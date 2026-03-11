use crate::errors::cudf_to_df;
use crate::expr::{columnar_value_to_cudf, cudf_to_columnar_value, expr_to_cudf_expr};
use arrow::array::RecordBatch;
use arrow_schema::{DataType, FieldRef, Schema};
use datafusion::common::DataFusionError;
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::expressions::BinaryExpr;
use datafusion::physical_expr::PhysicalExpr;
use datafusion_expr::Operator;
use delegate::delegate;
use libcudf_rs::{cast, cudf_binary_op, CuDFBinaryOp, CuDFColumnViewOrScalar};
use std::any::Any;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Debug, Clone, Eq)]
pub struct CuDFBinaryExpr {
    inner: BinaryExpr,

    left: Arc<dyn PhysicalExpr>,
    right: Arc<dyn PhysicalExpr>,
    op: CuDFBinaryOp,
}

impl PartialEq for CuDFBinaryExpr {
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }
}

impl Hash for CuDFBinaryExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl CuDFBinaryExpr {
    pub fn from_host(expr: BinaryExpr) -> Result<Self, DataFusionError> {
        let left = expr_to_cudf_expr(expr.left().as_ref())?;
        let right = expr_to_cudf_expr(expr.right().as_ref())?;
        let op = map_op(expr.op()).ok_or_else(|| {
            DataFusionError::NotImplemented(format!(
                "Operator {:?} is not supported by cuDF",
                expr.op()
            ))
        })?;
        Ok(Self {
            inner: expr,
            left,
            right,
            op,
        })
    }
}

impl Display for CuDFBinaryExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "CuDF")?;
        self.inner.fmt(f)
    }
}

impl PhysicalExpr for CuDFBinaryExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &RecordBatch) -> datafusion::common::Result<ColumnarValue> {
        let expected = self.data_type(batch.schema_ref())?;

        let mut lhs = columnar_value_to_cudf(self.left.evaluate(batch)?)?;
        let rhs = columnar_value_to_cudf(self.right.evaluate(batch)?)?;

        // cuDF Div computes floor(lhs_raw / rhs_raw) * 10^(s3-s1+s2). When s2+s3 > s1,
        // the floor step truncates before scaling is applied, e.g. 1.00/4.00 (s1=s2=2, s3=6):
        //   floor(100 / 400) * 10^6 = 0 -> wrong
        // Cast lhs to scale s2+s3 first so the cuDF factor becomes 10^0=1:
        //   cast lhs scale 2→8: raw 100 -> 100_000_000
        //   floor(100_000_000 / 400) * 10^0 = 250_000 at scale 6 = 0.250000 -> correct
        if self.op == CuDFBinaryOp::Div {
            let lhs_type = self.left.data_type(batch.schema_ref())?;
            let rhs_type = self.right.data_type(batch.schema_ref())?;
            if let (
                DataType::Decimal128(_, s1),
                DataType::Decimal128(_, s2),
                DataType::Decimal128(_, s3),
            ) = (&lhs_type, &rhs_type, &expected)
            {
                let target_scale = (*s2 as i32) + (*s3 as i32);
                let scale_shift = target_scale - (*s1 as i32);
                if scale_shift > 38 || target_scale > 38 {
                    return Err(DataFusionError::Internal(format!(
                        "decimal division pre-scaling overflow: shift={scale_shift}, target_scale={target_scale}"
                    )));
                }
                if scale_shift > 0 {
                    let lhs_view = match lhs {
                        CuDFColumnViewOrScalar::ColumnView(v) => v,
                        _ => {
                            return Err(DataFusionError::Internal(
                                "decimal division: expected column for lhs".into(),
                            ))
                        }
                    };
                    let cast_col = cast(&lhs_view, &DataType::Decimal128(38, target_scale as i8))
                        .map_err(cudf_to_df)?;
                    lhs = CuDFColumnViewOrScalar::ColumnView(cast_col.into_view());
                }
            }
        }

        let cudf_output_type = max_precision(&expected);
        let result = cudf_binary_op(lhs, rhs, self.op, &cudf_output_type).map_err(cudf_to_df)?;
        Ok(cudf_to_columnar_value(result))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> datafusion::common::Result<Arc<dyn PhysicalExpr>> {
        let expr = BinaryExpr::new(
            Arc::clone(&children[0]),
            *self.inner.op(),
            Arc::clone(&children[1]),
        );
        Ok(Arc::new(Self::from_host(expr)?))
    }

    delegate! {
        to self.inner {
            fn fmt_sql(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
            fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>>;
            fn data_type(&self, input_schema: &Schema) -> datafusion::common::Result<DataType>;
            fn return_field(&self, input_schema: &Schema) -> datafusion::common::Result<FieldRef>;
        }
    }
}

/// Normalise decimal precision to cuDF's fixed storage width; other types pass through unchanged.
fn max_precision(dt: &DataType) -> DataType {
    match dt {
        DataType::Decimal128(_, s) => DataType::Decimal128(38, *s),
        DataType::Decimal32(_, s) => DataType::Decimal32(9, *s),
        DataType::Decimal64(_, s) => DataType::Decimal64(18, *s),
        _ => dt.clone(),
    }
}

fn map_op(op: &Operator) -> Option<CuDFBinaryOp> {
    match op {
        // Comparison operators
        Operator::Eq => Some(CuDFBinaryOp::Equal),
        Operator::NotEq => Some(CuDFBinaryOp::NotEqual),
        Operator::Lt => Some(CuDFBinaryOp::Less),
        Operator::LtEq => Some(CuDFBinaryOp::LessEqual),
        Operator::Gt => Some(CuDFBinaryOp::Greater),
        Operator::GtEq => Some(CuDFBinaryOp::GreaterEqual),

        // Arithmetic operators
        Operator::Plus => Some(CuDFBinaryOp::Add),
        Operator::Minus => Some(CuDFBinaryOp::Sub),
        Operator::Multiply => Some(CuDFBinaryOp::Mul),
        Operator::Divide => Some(CuDFBinaryOp::Div),
        Operator::Modulo => Some(CuDFBinaryOp::Mod),

        // Logical operators (DataFusion And/Or are logical, not bitwise)
        Operator::And => Some(CuDFBinaryOp::LogicalAnd),
        Operator::Or => Some(CuDFBinaryOp::LogicalOr),

        // Null-aware comparison
        Operator::IsDistinctFrom => Some(CuDFBinaryOp::NullNotEquals),
        Operator::IsNotDistinctFrom => Some(CuDFBinaryOp::NullEquals),

        // Bitwise operators
        Operator::BitwiseAnd => Some(CuDFBinaryOp::BitwiseAnd),
        Operator::BitwiseOr => Some(CuDFBinaryOp::BitwiseOr),
        Operator::BitwiseXor => Some(CuDFBinaryOp::BitwiseXor),
        Operator::BitwiseShiftRight => Some(CuDFBinaryOp::ShiftRight),
        Operator::BitwiseShiftLeft => Some(CuDFBinaryOp::ShiftLeft),

        // Integer division
        Operator::IntegerDivide => Some(CuDFBinaryOp::FloorDiv),

        // Operators not supported by cuDF binary operations
        Operator::RegexMatch => None,
        Operator::RegexIMatch => None,
        Operator::RegexNotMatch => None,
        Operator::RegexNotIMatch => None,
        Operator::LikeMatch => None,
        Operator::ILikeMatch => None,
        Operator::NotLikeMatch => None,
        Operator::NotILikeMatch => None,
        Operator::StringConcat => None,

        // PostgreSQL-specific operators (not supported)
        Operator::AtArrow => None,
        Operator::ArrowAt => None,
        Operator::Arrow => None,
        Operator::LongArrow => None,
        Operator::HashArrow => None,
        Operator::HashLongArrow => None,
        Operator::AtAt => None,
        Operator::HashMinus => None,
        Operator::AtQuestion => None,
        Operator::Question => None,
        Operator::QuestionAnd => None,
        Operator::QuestionPipe => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_snapshot;
    use crate::test_utils::TestFramework;
    use datafusion::common::assert_contains;
    use std::error::Error;

    #[tokio::test]
    async fn test_binary_operations() -> Result<(), Box<dyn std::error::Error>> {
        let tf = TestFramework::new().await;

        let host_sql = r#"
            SELECT
                "MinTemp" + "MaxTemp" as addition,
                "MaxTemp" - "MinTemp" as subtraction,
                "MinTemp" * 2 as multiplication,
                "MaxTemp" / 2 as division,
                "Rainfall" % 10 as modulo,
                "MinTemp" = 12.2 as equal,
                "MinTemp" != 0.0 as not_equal,
                "MinTemp" < 15.0 as less_than,
                "MaxTemp" > 20.0 as greater_than,
                "MinTemp" <= 12.2 as less_equal,
                "MaxTemp" >= 24.3 as greater_equal,
                ("MaxTemp" - "MinTemp") * 2 as complex_expr
            FROM weather ORDER BY "MinTemp" LIMIT 1
        "#;
        let cudf_sql = format!(
            r#"
            SET datafusion.execution.target_partitions=1;
            SET cudf.enable=true;
            {host_sql}
        "#
        );

        let result = tf.execute(&cudf_sql).await?;
        assert_contains!(result.plan, "CuDF");
        assert_snapshot!(result.pretty_print, @"
        +----------+-------------+----------------+----------+--------+-------+-----------+-----------+--------------+------------+---------------+--------------+
        | addition | subtraction | multiplication | division | modulo | equal | not_equal | less_than | greater_than | less_equal | greater_equal | complex_expr |
        +----------+-------------+----------------+----------+--------+-------+-----------+-----------+--------------+------------+---------------+--------------+
        | 7.8      | 18.4        | -10.6          | 6.55     | 0.0    | false | true      | true      | false        | true       | false         | 36.8         |
        +----------+-------------+----------------+----------+--------+-------+-----------+-----------+--------------+------------+---------------+--------------+
        ");

        // Verify against host execution
        let host_result = tf.execute(host_sql).await?;
        assert_eq!(host_result.pretty_print, result.pretty_print);

        Ok(())
    }

    /// Decimal division where numerator < denominator would return 0 with raw cuDF Div.
    /// The pre-scaling fix in evaluate() must produce the same result as CPU execution.
    #[tokio::test]
    async fn test_decimal_division() -> Result<(), Box<dyn Error>> {
        let tf = TestFramework::new().await;
        let host_sql = r#"
            SELECT a / b as ratio
            FROM (VALUES
                (CAST(1.00 AS DECIMAL(10,2)), CAST(4.00 AS DECIMAL(10,2))),
                (CAST(3.00 AS DECIMAL(10,2)), CAST(4.00 AS DECIMAL(10,2))),
                (CAST(22.50 AS DECIMAL(10,2)), CAST(4.50 AS DECIMAL(10,2)))
            ) AS t(a, b)
            ORDER BY a
        "#;
        let cudf_sql = format!(
            "SET datafusion.execution.target_partitions=1; SET cudf.enable=true; {host_sql}"
        );
        let cudf = tf.execute(&cudf_sql).await?;
        let host = tf.execute(host_sql).await?;
        assert_contains!(cudf.plan, "CuDF");
        assert_eq!(host.pretty_print, cudf.pretty_print);
        Ok(())
    }
}
