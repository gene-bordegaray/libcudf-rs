use std::sync::Arc;

use arrow::array::ArrayData;

use crate::group_by::CuDFGroupBy;
use crate::{CuDFAstExpression, CuDFColumnView, CuDFScalar, CuDFStream, CuDFTableView};

/// Values retained until a stream-readiness event is no longer needed.
pub(crate) enum CuDFKeepAlive {
    Stream { _stream: Arc<CuDFStream> },
    TableView { _table: CuDFTableView },
    ColumnView { _column: CuDFColumnView },
    Scalar { _scalar: CuDFScalar },
    GroupBy { _group_by: CuDFGroupBy },
    AstExpression { _expression: CuDFAstExpression },
    ArrowData { _data: Arc<ArrayData> },
}
