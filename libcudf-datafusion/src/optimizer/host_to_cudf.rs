use crate::aggregate::try_as_cudf_aggregate;
use crate::optimizer::CuDFConfig;
use crate::physical::{
    is_cudf_plan, try_as_cudf_hash_join, CuDFCoalesceBatchesExec, CuDFFilterExec, CuDFHashJoinExec,
    CuDFLoadExec, CuDFProjectionExec, CuDFSortExec, CuDFUnloadExec,
};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::config::ConfigOptions;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_plan::aggregates::AggregateExec;
use datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::joins::HashJoinExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::ExecutionPlan;
use std::sync::Arc;

#[derive(Debug)]
pub struct HostToCuDFRule;

impl PhysicalOptimizerRule for HostToCuDFRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let Some(cudf_config) = config.extensions.get::<CuDFConfig>() else {
            return Ok(plan);
        };
        if !cudf_config.enable {
            return Ok(plan);
        }

        let result = plan.transform_up(|mut plan| {
            let mut cudf_node: Option<Arc<dyn ExecutionPlan>> = None;
            if let Some(node) = plan.as_any().downcast_ref::<FilterExec>() {
                cudf_node = Some(Arc::new(CuDFFilterExec::try_new(node.clone())?));
            }

            if let Some(node) = plan.as_any().downcast_ref::<ProjectionExec>() {
                cudf_node = Some(Arc::new(CuDFProjectionExec::try_new(node.clone())?));
            }

            if let Some(node) = plan.as_any().downcast_ref::<SortExec>() {
                cudf_node = Some(Arc::new(CuDFSortExec::try_new(node.clone())?));
            }

            if let Some(node) = plan.as_any().downcast_ref::<CoalesceBatchesExec>() {
                if is_cudf_plan(node.input().as_ref()) {
                    cudf_node = Some(Arc::new(CuDFCoalesceBatchesExec::from_input(
                        node.input().clone(),
                        cudf_config.batch_size,
                    )));
                }
            }

            if let Some(node) = plan.as_any().downcast_ref::<HashJoinExec>() {
                cudf_node = try_as_cudf_hash_join(node)?;
            }

            // TODO(#19): Multi-phase aggregate (Partial -> RepartitionExec -> Final) is not
            // supported. RepartitionExec is a CPU node; it strips CuDFColumnView wrappers from
            // partial-state batches, causing the downstream Final CuDFAggregateExec to fail.
            // Workaround: use with_target_partitions(1) to force AggregateMode::Single.
            // Fix: recognise the Partial->RepartitionExec->Final pattern in HostToCuDFRule and
            // either fuse into a single CuDFAggregateExec or bracket RepartitionExec with
            // CuDFUnloadExec / CuDFLoadExec to preserve column identity across the CPU boundary.
            if let Some(node) = plan.as_any().downcast_ref::<AggregateExec>() {
                cudf_node = try_as_cudf_aggregate(node)?;
            }

            let mut changed = false;
            if let Some(node) = cudf_node {
                plan = node;
                changed = true;
            }

            let plan_is_cudf = is_cudf_plan(plan.as_ref());
            let children = plan.children();
            let mut new_children: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(children.len());
            for (child_idx, child) in children.iter().enumerate() {
                let child_is_cudf = is_cudf_plan(child.as_ref());

                // The probe (right, index 1) child of CuDFHashJoinExec is uploaded in bulk
                // by the exec itself - do not insert CuDFLoadExec here.
                let is_probe_child = plan.as_any().is::<CuDFHashJoinExec>() && child_idx == 1;

                if plan_is_cudf
                    && !child_is_cudf
                    && !plan.as_any().is::<CuDFLoadExec>()
                    && !is_probe_child
                {
                    if !child.as_any().is::<CoalesceBatchesExec>() {
                        let child = Arc::new(CoalesceBatchesExec::new(
                            Arc::clone(child),
                            cudf_config.batch_size,
                        ));
                        new_children.push(Arc::new(CuDFLoadExec::try_new(child)?));
                    } else {
                        new_children.push(Arc::new(CuDFLoadExec::try_new(Arc::clone(child))?));
                    }
                    changed = true;
                    continue;
                }

                if !plan_is_cudf && child_is_cudf && !child.as_any().is::<CuDFUnloadExec>() {
                    let mut unload = if !child.as_any().is::<CuDFCoalesceBatchesExec>() {
                        let child = Arc::new(CuDFCoalesceBatchesExec::from_input(
                            Arc::clone(child),
                            cudf_config.batch_size,
                        ));
                        CuDFUnloadExec::new(child)
                    } else {
                        CuDFUnloadExec::new(Arc::clone(child))
                    };
                    // Aggregations will expect a specific schema in, which is the one that was
                    // established while the node was placed there. As we are dealing with type
                    // incompatibilities in CuDF, we are tweaking the schema we return, and
                    // therefore, we might need to manually force a cast.
                    if let Some(agg) = plan.as_any().downcast_ref::<AggregateExec>() {
                        unload = unload.with_target_schema(Arc::clone(&agg.input_schema))
                    }
                    new_children.push(Arc::new(unload));
                    changed = true;
                    continue;
                }

                new_children.push(Arc::clone(child));
            }

            if changed {
                Ok(Transformed::yes(plan.with_new_children(new_children)?))
            } else {
                Ok(Transformed::no(plan))
            }
        })?;

        if is_cudf_plan(result.data.as_ref()) {
            Ok(Arc::new(CuDFUnloadExec::new(result.data)))
        } else {
            Ok(result.data)
        }
    }

    fn name(&self) -> &str {
        "HostToCuDFRule"
    }

    fn schema_check(&self) -> bool {
        false
    }
}
