use arrow::array::RecordBatch;
use arrow_schema::{DataType, FieldRef, Schema};
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::PhysicalExpr;
use datafusion_physical_plan::expressions::Column;
use delegate::delegate;
use std::any::Any;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct CuDFColumnExpr {
    inner: Column,
}

impl CuDFColumnExpr {
    pub fn from_host(inner: Column) -> Self {
        Self { inner }
    }
}

impl Display for CuDFColumnExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "CuDF")?;
        self.inner.fmt(f)
    }
}

impl PhysicalExpr for CuDFColumnExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &RecordBatch) -> datafusion::common::Result<ColumnarValue> {
        self.inner.evaluate(batch)
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> datafusion::common::Result<Arc<dyn PhysicalExpr>> {
        Ok(self)
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

#[cfg(test)]
mod tests {
    use crate::assert_snapshot;
    use crate::test_utils::TestFramework;
    use datafusion::common::assert_contains;

    // Ignored: ORDER BY a non-projected column produces wrong results due to a bug in
    // extract_sort_params which ignores the column index in the sort expression and always
    // sorts by column 0. https://github.com/gene-bordegaray/libcudf-rs/issues/23
    #[ignore]
    #[tokio::test]
    async fn test_column_in_expressions() -> Result<(), Box<dyn std::error::Error>> {
        let tf = TestFramework::new().await;

        let host_sql = r#"
            SELECT
                "MinTemp" + "MaxTemp" as sum_temps,
                "MinTemp" * 2 as doubled_min,
                "MaxTemp" - 10.0 as offset_max
            FROM weather
            ORDER BY "MinTemp"
            LIMIT 2
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
        assert_snapshot!(result.pretty_print, @"");

        let host_result = tf.execute(host_sql).await?;
        assert_eq!(host_result.pretty_print, result.pretty_print);

        Ok(())
    }
}
