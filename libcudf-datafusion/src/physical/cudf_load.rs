use crate::errors::cudf_to_df;
use arrow::array::{Array, RecordBatch};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Schema, SchemaRef};
use datafusion::common::{exec_err, plan_err, ScalarValue};
use datafusion::error::DataFusionError;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures_util::stream::StreamExt;
use libcudf_rs::{is_cudf_array, CuDFColumn};
use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;

#[derive(Debug)]
pub struct CuDFLoadExec {
    input: Arc<dyn ExecutionPlan>,

    properties: PlanProperties,
}

impl CuDFLoadExec {
    pub fn try_new(input: Arc<dyn ExecutionPlan>) -> Result<Self, DataFusionError> {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(cudf_schema_compatibility_map(input.schema())),
            input.properties().partitioning.clone(),
            input.properties().emission_type,
            input.properties().boundedness,
        );
        Ok(Self { input, properties })
    }
}

impl DisplayAs for CuDFLoadExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "CuDFLoadExec")
    }
}

impl ExecutionPlan for CuDFLoadExec {
    fn name(&self) -> &str {
        "CuDFLoadExec"
    }

    // TODO: implement partition_statistics by delegating to self.input.partition_statistics().
    // The schema changes (cudf_schema_compatibility_map remaps types), but row counts and
    // column-level statistics remain valid and should be preserved.

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return plan_err!(
                "CuDFLoadExec expects exactly 1 child, {} where provided",
                children.len()
            );
        }
        let input = Arc::clone(&children[0]);
        Ok(Arc::new(Self::try_new(input)?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::common::Result<SendableRecordBatchStream> {
        let host_stream = self.input.execute(partition, context)?;
        let target_schema = self.schema();

        let cudf_stream = host_stream.map(move |batch_or_err| {
            let batch = match batch_or_err {
                Ok(batch) => cast_to_target_schema(batch, Arc::clone(&target_schema))?,
                Err(err) => return Err(err),
            };

            // TODO(#20): replace per-column upload with a single CuDFTable::from_arrow_host call
            // see https://github.com/gene-bordegaray/libcudf-rs/issues/20
            let original_cols = batch.columns();
            let mut cudf_cols: Vec<Arc<dyn Array>> = Vec::with_capacity(original_cols.len());
            for original_col in original_cols {
                if is_cudf_array(original_col) {
                    return exec_err!(
                        "Cannot move RecordBatch from host to CuDF: a column is already a CuDF array"
                    );
                }
                let col = CuDFColumn::from_arrow_host(original_col).map_err(cudf_to_df)?;
                cudf_cols.push(Arc::new(col.into_view()));
            }

            Ok(RecordBatch::try_new(batch.schema(), cudf_cols)?)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            cudf_stream,
        )))
    }
}

/// Converts Arrow scalar types that cuDF doesn't support (e.g. `Utf8View`) into their
/// cuDF-compatible equivalents.
pub(crate) fn normalize_scalar_for_cudf(value: ScalarValue) -> ScalarValue {
    match value {
        ScalarValue::Utf8View(s) => ScalarValue::Utf8(s),
        other => other,
    }
}

/// Maps an Arrow schema to the types cuDF will actually produce.
///
/// - `Utf8View -> Utf8`: cuDF's `TYPE_STRING` has no view variant.
/// - `Decimal{32,64,128}(p, s) -> Decimal{32,64,128}(max_p, s)`: cuDF uses fixed precision
///   based on storage width (9/18/38). Scale is preserved.
pub(crate) fn cudf_schema_compatibility_map(schema: SchemaRef) -> SchemaRef {
    let mut new_fields = Vec::with_capacity(schema.fields.len());

    for field in schema.fields() {
        let field = match field.data_type() {
            DataType::Utf8View => FieldRef::new(Field::new(
                field.name(),
                DataType::Utf8,
                field.is_nullable(),
            )),
            DataType::Decimal32(_, s) => FieldRef::new(Field::new(
                field.name(),
                DataType::Decimal32(9, *s), // max precision for 32-bit
                field.is_nullable(),
            )),
            DataType::Decimal64(_, s) => FieldRef::new(Field::new(
                field.name(),
                DataType::Decimal64(18, *s), // max precision for 64-bit
                field.is_nullable(),
            )),
            DataType::Decimal128(_, s) => FieldRef::new(Field::new(
                field.name(),
                DataType::Decimal128(38, *s), // max precision for 128-bit
                field.is_nullable(),
            )),
            _ => Arc::clone(field),
        };
        new_fields.push(field);
    }

    SchemaRef::new(Schema::new(new_fields))
}

pub(crate) fn cast_to_target_schema(
    batch: RecordBatch,
    target_schema: SchemaRef,
) -> Result<RecordBatch, ArrowError> {
    let columns = batch
        .columns()
        .iter()
        .zip(target_schema.fields())
        .map(|(col, field)| arrow::compute::cast(col, field.data_type()))
        .collect::<Result<Vec<_>, _>>()?;

    RecordBatch::try_new(target_schema, columns)
}
