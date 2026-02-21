use crate::aggregate::op::udf::CuDFAggregateUDF;
use crate::aggregate::CuDFAggregationOp;
use crate::errors::cudf_to_df;
use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use datafusion::common::{exec_err, internal_err};
use datafusion::error::Result;
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream};
use datafusion::physical_expr::PhysicalExpr;
use datafusion_physical_plan::aggregates::{
    evaluate_group_by, evaluate_many, AggregateMode, PhysicalGroupBy,
};
use datafusion_physical_plan::udaf::AggregateFunctionExpr;
use futures::{ready, StreamExt};
use libcudf_rs::{CuDFColumn, CuDFColumnView, CuDFGroupBy, CuDFTable, CuDFTableView};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

enum StreamState {
    ReadingInput,
    ProducingOutput,
    Done,
}

/// Maps each aggregation op to its slice of the flat state_columns array.
///
/// Built once at construction from `num_state_columns()`.
///
/// Example for `[SUM, AVG, MAX]`:
/// ```text
///   SUM -> (0, 1), AVG -> (1, 2), MAX -> (3, 1)
/// ```
struct ColumnMapping {
    ranges: Vec<(usize, usize)>, // (start, count) per op
}

/// Running O(G) state: unique group keys + intermediate state columns.
struct RunningState {
    keys: CuDFTable,
    state_columns: Vec<CuDFColumn>, // flat, indexed via ColumnMapping
}

/// GPU-accelerated GROUP BY aggregation stream using rolling merge.
///
/// Each input batch is aggregated on the GPU, then immediately merged into a
/// running state that holds only the unique group keys and their intermediate
/// state columns.
///
/// After all input is consumed, a single output batch is produced by either
/// finalizing the state (Single/Final modes) or emitting raw state columns
/// (Partial mode).
///
/// # Performance
///
/// - O(G) memory where G is the number of groups, regardless of how many input
///   batches.
pub struct Stream {
    input: SendableRecordBatchStream,
    output_schema: SchemaRef,
    mode: AggregateMode,
    group_by: PhysicalGroupBy,
    /// DataFusion expressions -> used for schema metadata (e.g., `state_fields()`).
    aggregate_expr: Vec<Arc<AggregateFunctionExpr>>,
    /// Physical expressions that extract each op's argument columns from a batch.
    aggregate_args: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    /// GPU aggregation implementations (one per aggregate function).
    aggregate_ops: Vec<Arc<dyn CuDFAggregationOp>>,
    column_mapping: ColumnMapping,
    state: StreamState,
    running: Option<RunningState>,
}

impl Stream {
    pub fn new(
        input: SendableRecordBatchStream,
        output_schema: SchemaRef,
        mode: AggregateMode,
        group_by: PhysicalGroupBy,
        aggregate_expr: Vec<Arc<AggregateFunctionExpr>>,
    ) -> Self {
        let aggregate_args = aggregate_expr
            .iter()
            .map(|x| x.expressions())
            .collect::<Vec<_>>();

        // Extract the GPU op from each CuDFAggregateUDF wrapper.
        // Safe to unwrap: the optimizer only creates CuDFAggregateExec when all
        // aggregate functions are backed by CuDFAggregateUDF.
        let aggregate_ops = aggregate_expr
            .iter()
            .map(|expr| {
                expr.fun()
                    .inner()
                    .as_any()
                    .downcast_ref::<CuDFAggregateUDF>()
                    .expect("aggregate expr should be CuDFAggregateUDF")
                    .gpu()
                    .clone()
            })
            .collect::<Vec<_>>();

        let column_mapping = {
            let mut offset = 0;
            let ranges = aggregate_ops
                .iter()
                .map(|op| {
                    let count = op.num_state_columns();
                    let start = offset;
                    offset += count;
                    (start, count)
                })
                .collect();
            ColumnMapping { ranges }
        };

        Self {
            input,
            output_schema,
            mode,
            group_by,
            aggregate_expr,
            aggregate_args,
            aggregate_ops,
            column_mapping,
            state: StreamState::ReadingInput,
            running: None,
        }
    }

    /// Process a single input batch: aggregate it and merge into running state.
    fn process_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let group_by = self.evaluate_batch_groups(batch)?;
        let evaluated_args = self.evaluate_batch_arguments(batch)?;

        // Build requests based on mode
        let requests = self.build_batch_requests(evaluated_args)?;

        let (batch_keys, batch_results) = group_by.aggregate(&requests).map_err(cudf_to_df)?;
        let mut batch_state_columns = batch_results.into_iter().flatten().collect();

        // Normalize partial state column types so they match merge_requests output types.
        if !matches!(
            self.mode,
            AggregateMode::Final | AggregateMode::FinalPartitioned
        ) {
            batch_state_columns = self.normalize_partial_state(batch_state_columns)?;
        }

        self.merge_into_running(batch_keys, batch_state_columns)
    }

    /// Dispatch `normalize_partial_state` for each op over the flat state column vec.
    fn normalize_partial_state(&self, cols: Vec<CuDFColumn>) -> Result<Vec<CuDFColumn>> {
        let mut result = Vec::with_capacity(cols.len());
        let mut col_iter = cols.into_iter();
        for op_idx in 0..self.aggregate_ops.len() {
            let (_, count) = self.column_mapping.ranges[op_idx];
            let op_cols: Vec<CuDFColumn> = col_iter.by_ref().take(count).collect();
            result.extend(self.aggregate_ops[op_idx].normalize_partial_state(op_cols)?);
        }
        Ok(result)
    }

    /// Build aggregation requests for a single batch based on mode.
    ///
    /// - Single/Partial/SinglePartitioned: use `partial_requests` (input is raw data)
    /// - Final/FinalPartitioned: use `merge_requests` (input is partial state)
    fn build_batch_requests(
        &self,
        evaluated_args: Vec<Vec<CuDFColumnView>>,
    ) -> Result<Vec<libcudf_rs::AggregationRequest>> {
        let mut requests = Vec::new();

        let use_merge = matches!(
            self.mode,
            AggregateMode::Final | AggregateMode::FinalPartitioned
        );

        for (op, args) in self.aggregate_ops.iter().zip(evaluated_args) {
            let op_requests = if use_merge {
                op.merge_requests(&args)?
            } else {
                op.partial_requests(&args)?
            };
            requests.extend(op_requests);
        }

        Ok(requests)
    }

    /// Merge new batch results into the running state.
    ///
    /// If no running state exists, stores the new results directly.
    /// Otherwise, concatenates running + new, then re-aggregates with merge_requests.
    fn merge_into_running(
        &mut self,
        new_keys: CuDFTable,
        new_state_columns: Vec<CuDFColumn>,
    ) -> Result<()> {
        let Some(running) = self.running.take() else {
            self.running = Some(RunningState {
                keys: new_keys,
                state_columns: new_state_columns,
            });
            return Ok(());
        };

        // Concat keys
        let combined_keys = CuDFTable::concat(vec![running.keys.into_view(), new_keys.into_view()])
            .map_err(cudf_to_df)?;

        // Concat each state column pair
        let mut combined_state_columns = Vec::with_capacity(running.state_columns.len());
        for (run_col, new_col) in running
            .state_columns
            .into_iter()
            .zip(new_state_columns.into_iter())
        {
            let combined = CuDFColumn::concat(vec![run_col.into_view(), new_col.into_view()])
                .map_err(cudf_to_df)?;
            combined_state_columns.push(combined);
        }

        // Convert to views for merge requests
        let combined_views: Vec<CuDFColumnView> = combined_state_columns
            .into_iter()
            .map(|col| col.into_view())
            .collect();

        // Build merge requests
        let mut requests = Vec::new();
        for (op_idx, op) in self.aggregate_ops.iter().enumerate() {
            let (start, count) = self.column_mapping.ranges[op_idx];
            let state_views: Vec<CuDFColumnView> = combined_views[start..start + count].to_vec();
            requests.extend(op.merge_requests(&state_views)?);
        }

        // Re-aggregate
        let group_by = CuDFGroupBy::from_table_view(combined_keys.into_view());
        let (merged_keys, merged_results) = group_by.aggregate(&requests).map_err(cudf_to_df)?;
        let merged_state_columns = merged_results.into_iter().flatten().collect();

        self.running = Some(RunningState {
            keys: merged_keys,
            state_columns: merged_state_columns,
        });

        Ok(())
    }

    /// Build the final output RecordBatch from the running state.
    fn build_output(&mut self) -> Result<Option<RecordBatch>> {
        let Some(running) = self.running.take() else {
            return Ok(None);
        };

        let key_columns = running.keys.into_columns();
        let mut arrays: Vec<ArrayRef> =
            Vec::with_capacity(key_columns.len() + self.aggregate_expr.len());

        for col in key_columns {
            arrays.push(Arc::new(col.into_view()));
        }

        let state_views: Vec<CuDFColumnView> = running
            .state_columns
            .into_iter()
            .map(|c| c.into_view())
            .collect();

        let is_partial = matches!(self.mode, AggregateMode::Partial);

        for (op_idx, op) in self.aggregate_ops.iter().enumerate() {
            let (start, count) = self.column_mapping.ranges[op_idx];
            let state_views: Vec<CuDFColumnView> = state_views[start..start + count].to_vec();

            if is_partial {
                // Partial mode: emit raw state columns, cast to match state_fields schema
                let state_fields = self.aggregate_expr[op_idx].state_fields()?;
                for (col_idx, view) in state_views.into_iter().enumerate() {
                    let target_type = state_fields[col_idx].data_type();
                    if view.data_type() != target_type {
                        let casted = libcudf_rs::cast(&view, target_type).map_err(cudf_to_df)?;
                        arrays.push(Arc::new(casted.into_view()));
                    } else {
                        arrays.push(Arc::new(view));
                    }
                }
            } else {
                // Single/Final/FinalPartitioned/SinglePartitioned: finalize
                let finalized = op.finalize(&state_views)?;
                arrays.push(Arc::new(finalized));
            }
        }

        Ok(Some(RecordBatch::try_new(
            self.output_schema.clone(),
            arrays,
        )?))
    }

    /// Evaluate GROUP BY expressions on a batch and wrap the resulting key
    /// columns into a [`CuDFGroupBy`] for GPU aggregation.
    fn evaluate_batch_groups(&self, batch: &RecordBatch) -> Result<CuDFGroupBy> {
        let grouping_sets = evaluate_group_by(&self.group_by, batch)?;

        if grouping_sets.len() != 1 {
            return exec_err!("Expected single grouping set, got {}", grouping_sets.len());
        }

        let group = &grouping_sets[0];
        let column_views = group
            .iter()
            .map(|arr| arr.as_any().downcast_ref::<CuDFColumnView>().unwrap())
            .cloned()
            .collect::<Vec<_>>();

        let table_view = CuDFTableView::from_column_views(column_views).map_err(cudf_to_df)?;

        Ok(CuDFGroupBy::from_table_view(table_view))
    }

    /// Evaluate each aggregate function's argument expressions on a batch,
    /// returning GPU column views suitable for `partial_requests` / `merge_requests`.
    fn evaluate_batch_arguments(&self, batch: &RecordBatch) -> Result<Vec<Vec<CuDFColumnView>>> {
        let evaluated_arguments = evaluate_many(&self.aggregate_args, batch)?;

        evaluated_arguments
            .iter()
            .map(|args| {
                args.iter()
                    .map(|arg| {
                        let Some(view) = arg.as_any().downcast_ref::<CuDFColumnView>() else {
                            return internal_err!("Expected Array to be of type CuDFColumnView");
                        };
                        Ok(view.clone())
                    })
                    .collect()
            })
            .collect()
    }
}

impl futures::Stream for Stream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &self.state {
                StreamState::ReadingInput => match ready!(self.input.poll_next_unpin(cx)) {
                    None => {
                        self.state = StreamState::ProducingOutput;
                    }
                    Some(Err(e)) => return Poll::Ready(Some(Err(e))),
                    Some(Ok(batch)) => {
                        self.process_batch(&batch)?;
                    }
                },
                StreamState::ProducingOutput => {
                    let output = self.build_output()?;
                    self.state = StreamState::Done;
                    return match output {
                        Some(batch) => Poll::Ready(Some(Ok(batch))),
                        None => Poll::Ready(None),
                    };
                }
                StreamState::Done => return Poll::Ready(None),
            }
        }
    }
}

impl RecordBatchStream for Stream {
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }
}
