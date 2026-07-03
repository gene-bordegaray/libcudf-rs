use crate::column::CuDFColumn;
use crate::data_type::arrow_type_to_cudf_data_type;
use crate::deferred_operation::deferred;
use crate::execution_policy::OperationLaunch;
use crate::{CuDFColumnView, CuDFError, CuDFExecutionContext, CuDFOperation, CuDFScalar};
use arrow_schema::{ArrowError, DataType};
use libcudf_sys::ffi;

/// A cuDF column view or scalar value.
///
/// This is the operand/result shape for cuDF APIs that accept either a column
/// or a scalar value. Binary operations require at least one operand to be a
/// column.
pub enum CuDFColumnViewOrScalar {
    /// A column view containing one value per row.
    ColumnView(CuDFColumnView),
    /// A single scalar value.
    Scalar(CuDFScalar),
}

impl CuDFColumnViewOrScalar {
    /// Create a deferred operation that applies a cuDF binary operation.
    ///
    /// Calling this method only captures the operands and options. cuDF work is
    /// not submitted until the returned operation is passed to
    /// [`CuDFExecutionContext::execute`](crate::CuDFExecutionContext::execute).
    /// Execution then waits for both operands to be ready on the target context
    /// stream before launching the binary operation. At least one operand must
    /// be a column; scalar-scalar binary operations are rejected by the cuDF
    /// binding used here.
    ///
    /// # Errors
    ///
    /// Execution returns an error if:
    /// - `output_type` is not supported by cuDF
    /// - the operands are incompatible with `op`
    /// - both operands are scalars
    /// - cuDF cannot allocate the output column
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow::array::Int32Array;
    /// use arrow_schema::DataType;
    /// use libcudf_rs::{CuDFBinaryOp, CuDFColumn, CuDFColumnViewOrScalar, CuDFExecutionContext};
    ///
    /// let lhs = Int32Array::from(vec![1, 2, 3]);
    /// let rhs = Int32Array::from(vec![10, 20, 30]);
    ///
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let lhs = ctx.execute(CuDFColumn::from_arrow_host(&lhs))?.into_view();
    /// let rhs = ctx.execute(CuDFColumn::from_arrow_host(&rhs))?.into_view();
    ///
    /// let result = ctx.execute(
    ///     CuDFColumnViewOrScalar::ColumnView(lhs).binary_op(
    ///         CuDFColumnViewOrScalar::ColumnView(rhs),
    ///         CuDFBinaryOp::Add,
    ///         &DataType::Int32,
    ///     ),
    /// )?;
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn binary_op<'a>(
        self,
        right: Self,
        op: CuDFBinaryOp,
        output_type: &'a DataType,
    ) -> impl CuDFOperation<Output = Self> + 'a {
        deferred(move |ctx| binary_op_on_context(ctx, self, right, op, output_type))
    }
}

impl From<CuDFColumnView> for CuDFColumnViewOrScalar {
    fn from(col: CuDFColumnView) -> Self {
        Self::ColumnView(col)
    }
}

impl From<CuDFScalar> for CuDFColumnViewOrScalar {
    fn from(scalar: CuDFScalar) -> Self {
        Self::Scalar(scalar)
    }
}

/// A binary operator supported by cuDF column/scalar operations.
///
/// This maps to cuDF's `binary_operator` enum. The generic PTX-backed
/// operator is listed for enum parity, but this wrapper does not expose the
/// extra UDF input needed to execute it through [`CuDFColumnViewOrScalar::binary_op`].
///
/// [`CuDFColumnViewOrScalar::binary_op`]: crate::CuDFColumnViewOrScalar::binary_op
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CuDFBinaryOp {
    /// operator +
    Add = 0,
    /// operator -
    Sub = 1,
    /// operator *
    Mul = 2,
    /// operator / using common type of lhs and rhs
    Div = 3,
    /// operator / after promoting type to floating point
    TrueDiv = 4,
    /// operator // (floor division)
    FloorDiv = 5,
    /// operator %
    Mod = 6,
    /// positive modulo operator
    PMod = 7,
    /// operator % but following Python's sign rules for negatives
    PyMod = 8,
    /// lhs ^ rhs
    Pow = 9,
    /// int ^ int, used to avoid floating point precision loss
    IntPow = 10,
    /// logarithm to the base
    LogBase = 11,
    /// 2-argument arctangent
    Atan2 = 12,
    /// operator <<
    ShiftLeft = 13,
    /// operator >>
    ShiftRight = 14,
    /// operator >>> (logical right shift, from Java)
    ShiftRightUnsigned = 15,
    /// operator &
    BitwiseAnd = 16,
    /// operator |
    BitwiseOr = 17,
    /// operator ^
    BitwiseXor = 18,
    /// operator &&
    LogicalAnd = 19,
    /// operator ||
    LogicalOr = 20,
    /// operator ==
    Equal = 21,
    /// operator !=
    NotEqual = 22,
    /// operator <
    Less = 23,
    /// operator >
    Greater = 24,
    /// operator <=
    LessEqual = 25,
    /// operator >=
    GreaterEqual = 26,
    /// Returns true when both operands are null; false when one is null;
    /// the result of equality when both are non-null
    NullEquals = 27,
    /// Returns false when both operands are null; true when one is null;
    /// the result of inequality when both are non-null
    NullNotEquals = 28,
    /// Returns max of operands when both are non-null; returns the non-null
    /// operand when one is null; or invalid when both are null
    NullMax = 29,
    /// Returns min of operands when both are non-null; returns the non-null
    /// operand when one is null; or invalid when both are null
    NullMin = 30,
    /// Generic binary operator generated from PTX code.
    ///
    /// This variant cannot be executed through the current high-level binary
    /// operation API because that API has no PTX/code input.
    GenericBinary = 31,
    /// operator && with Spark rules
    NullLogicalAnd = 32,
    /// operator || with Spark rules
    NullLogicalOr = 33,
    /// invalid operation sentinel
    InvalidBinary = 34,
}

pub(crate) fn binary_op_on_context(
    ctx: &CuDFExecutionContext,
    left: CuDFColumnViewOrScalar,
    right: CuDFColumnViewOrScalar,
    op: CuDFBinaryOp,
    output_type: &DataType,
) -> Result<CuDFColumnViewOrScalar, CuDFError> {
    let Some(dt) = arrow_type_to_cudf_data_type(output_type) else {
        return Err(ArrowError::NotYetImplemented(format!(
            "Output type {output_type} not supported in CuDF"
        )))?;
    };
    let op = cudf_binary_operator(op)?;
    let mut launch = crate::execution_policy::launch(ctx)?;
    wait_operand(&mut launch, &left)?;
    wait_operand(&mut launch, &right)?;

    let result = match (left, right) {
        (CuDFColumnViewOrScalar::ColumnView(lhs), CuDFColumnViewOrScalar::ColumnView(rhs)) => {
            ffi::binary_operation_col_col(
                lhs.inner(),
                rhs.inner(),
                op,
                &dt,
                launch.stream()?,
                launch.resource(),
            )
        }
        (CuDFColumnViewOrScalar::ColumnView(lhs), CuDFColumnViewOrScalar::Scalar(rhs)) => {
            ffi::binary_operation_col_scalar(
                lhs.inner(),
                rhs.inner(),
                op,
                &dt,
                launch.stream()?,
                launch.resource(),
            )
        }
        (CuDFColumnViewOrScalar::Scalar(lhs), CuDFColumnViewOrScalar::ColumnView(rhs)) => {
            ffi::binary_operation_scalar_col(
                lhs.inner(),
                rhs.inner(),
                op,
                &dt,
                launch.stream()?,
                launch.resource(),
            )
        }
        (CuDFColumnViewOrScalar::Scalar(_), CuDFColumnViewOrScalar::Scalar(_)) => {
            return Err(ArrowError::InvalidArgumentError(
                "cuDF binary operation requires at least one column operand".to_string(),
            ))?
        }
    }?;
    let column = launch.ready_column(CuDFColumn::from_inner(result))?;
    Ok(CuDFColumnViewOrScalar::ColumnView(column.into_view()))
}

fn cudf_binary_operator(op: CuDFBinaryOp) -> Result<i32, CuDFError> {
    match op {
        CuDFBinaryOp::GenericBinary => Err(ArrowError::NotYetImplemented(
            "GenericBinary requires PTX input that is not exposed by this API".to_string(),
        ))?,
        CuDFBinaryOp::InvalidBinary => Err(ArrowError::InvalidArgumentError(
            "InvalidBinary is not an executable cuDF binary operator".to_string(),
        ))?,
        _ => Ok(op as i32),
    }
}

fn wait_operand(
    launch: &mut OperationLaunch<'_>,
    value: &CuDFColumnViewOrScalar,
) -> Result<(), CuDFError> {
    match value {
        CuDFColumnViewOrScalar::ColumnView(column) => launch.wait_column(column),
        CuDFColumnViewOrScalar::Scalar(scalar) => launch.wait_scalar(scalar),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Decimal128Array};
    use arrow::datatypes::DataType;

    #[test]
    fn test_decimal_addition() -> Result<(), Box<dyn std::error::Error>> {
        // Create two decimal columns with scale=2 (e.g., currency values)
        let values1 = vec![12345i128, 67890i128, 11111i128]; // 123.45, 678.90, 111.11
        let values2 = vec![54321i128, 98765i128, 22222i128]; // 543.21, 987.65, 222.22

        let array1 = Decimal128Array::from(values1).with_precision_and_scale(38, 2)?;
        let array2 = Decimal128Array::from(values2).with_precision_and_scale(38, 2)?;

        let col1 = crate::execute_cudf(CuDFColumn::from_arrow_host(&array1))?;
        let col2 = crate::execute_cudf(CuDFColumn::from_arrow_host(&array2))?;

        // Add the two decimal columns
        let result = crate::execute_cudf(
            CuDFColumnViewOrScalar::ColumnView(col1.into_view()).binary_op(
                CuDFColumnViewOrScalar::ColumnView(col2.into_view()),
                CuDFBinaryOp::Add,
                &DataType::Decimal128(38, 2),
            ),
        )?;

        // Convert result back to Arrow
        let result_col = match result {
            CuDFColumnViewOrScalar::ColumnView(view) => view,
            _ => panic!("Expected column view"),
        };

        let result_array = crate::execute_cudf(result_col.to_arrow_host())?;
        let result_decimal = result_array
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("Expected Decimal128Array");

        // Verify results: 123.45 + 543.21 = 666.66, etc.
        assert_eq!(result_decimal.len(), 3);
        assert_eq!(result_decimal.value(0), 66666); // 666.66 with scale=2
        assert_eq!(result_decimal.value(1), 166655); // 1666.55 with scale=2
        assert_eq!(result_decimal.value(2), 33333); // 333.33 with scale=2

        Ok(())
    }

    #[test]
    fn test_decimal_multiplication() -> Result<(), Box<dyn std::error::Error>> {
        // Create decimal columns: price * quantity
        let prices = vec![1050i128, 2500i128]; // 10.50, 25.00 with scale=2
        let quantities = vec![3i128, 5i128]; // 3, 5 with scale=0

        let price_array = Decimal128Array::from(prices).with_precision_and_scale(38, 2)?;
        let qty_array = Decimal128Array::from(quantities).with_precision_and_scale(38, 0)?;

        let col_price = crate::execute_cudf(CuDFColumn::from_arrow_host(&price_array))?;
        let col_qty = crate::execute_cudf(CuDFColumn::from_arrow_host(&qty_array))?;

        // Multiply: result should have scale = 2 + 0 = 2
        let result = crate::execute_cudf(
            CuDFColumnViewOrScalar::ColumnView(col_price.into_view()).binary_op(
                CuDFColumnViewOrScalar::ColumnView(col_qty.into_view()),
                CuDFBinaryOp::Mul,
                &DataType::Decimal128(38, 2),
            ),
        )?;

        let result_col = match result {
            CuDFColumnViewOrScalar::ColumnView(view) => view,
            _ => panic!("Expected column view"),
        };

        let result_array = crate::execute_cudf(result_col.to_arrow_host())?;
        let result_decimal = result_array
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("Expected Decimal128Array");

        // Verify: 10.50 * 3 = 31.50, 25.00 * 5 = 125.00
        assert_eq!(result_decimal.value(0), 3150); // 31.50 with scale=2
        assert_eq!(result_decimal.value(1), 12500); // 125.00 with scale=2

        Ok(())
    }
}
