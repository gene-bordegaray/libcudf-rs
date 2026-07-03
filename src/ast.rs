use crate::{CuDFError, CuDFScalar};
use arrow::error::ArrowError;
use cxx::UniquePtr;
use libcudf_sys::{ffi, AstOperator, AstTableReference};
use std::fmt;
use std::sync::Arc;

/// Opaque node handle inside a [`CuDFAstExpression`] tree.
///
/// Node handles are returned by [`CuDFAstExpression`] builder methods and can
/// be used as inputs to later operation nodes in the same expression tree.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CuDFAstNode(usize);

impl CuDFAstNode {
    fn from_raw(index: usize) -> Self {
        Self(index)
    }

    fn into_raw(self) -> usize {
        self.0
    }
}

/// Table referenced by a cuDF AST column node.
///
/// Join predicates evaluate against two input tables, while other cuDF AST
/// consumers can also evaluate expressions against an output table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CuDFAstTableReference {
    /// Column index in the left table.
    Left,
    /// Column index in the right table.
    Right,
    /// Column index in the output table.
    Output,
}

impl CuDFAstTableReference {
    fn into_sys(self) -> AstTableReference {
        match self {
            Self::Left => AstTableReference::Left,
            Self::Right => AstTableReference::Right,
            Self::Output => AstTableReference::Output,
        }
    }
}

/// Operators supported by [`CuDFAstExpression`].
///
/// cuDF validates whether an operator is valid for the node arity and input
/// types when the node is added to an expression tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CuDFAstOperator {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Equality comparison.
    Equal,
    /// Non-equality comparison.
    NotEqual,
    /// Less-than comparison.
    Less,
    /// Greater-than comparison.
    Greater,
    /// Less-than-or-equal comparison.
    LessEqual,
    /// Greater-than-or-equal comparison.
    GreaterEqual,
    /// Logical AND.
    LogicalAnd,
    /// Null-aware logical AND.
    NullLogicalAnd,
    /// Logical OR.
    LogicalOr,
    /// Null-aware logical OR.
    NullLogicalOr,
    /// Modulo.
    Mod,
    /// Null-aware equality comparison.
    NullEqual,
    /// Null check.
    IsNull,
    /// Logical NOT.
    Not,
    /// Cast to int64.
    CastToInt64,
    /// Cast to uint64.
    CastToUint64,
    /// Cast to float64.
    CastToFloat64,
}

impl CuDFAstOperator {
    fn into_sys(self) -> AstOperator {
        match self {
            Self::Add => AstOperator::Add,
            Self::Sub => AstOperator::Sub,
            Self::Mul => AstOperator::Mul,
            Self::Div => AstOperator::Div,
            Self::Equal => AstOperator::Equal,
            Self::NotEqual => AstOperator::NotEqual,
            Self::Less => AstOperator::Less,
            Self::Greater => AstOperator::Greater,
            Self::LessEqual => AstOperator::LessEqual,
            Self::GreaterEqual => AstOperator::GreaterEqual,
            Self::LogicalAnd => AstOperator::LogicalAnd,
            Self::NullLogicalAnd => AstOperator::NullLogicalAnd,
            Self::LogicalOr => AstOperator::LogicalOr,
            Self::NullLogicalOr => AstOperator::NullLogicalOr,
            Self::Mod => AstOperator::Mod,
            Self::NullEqual => AstOperator::NullEqual,
            Self::IsNull => AstOperator::IsNull,
            Self::Not => AstOperator::Not,
            Self::CastToInt64 => AstOperator::CastToInt64,
            Self::CastToUint64 => AstOperator::CastToUint64,
            Self::CastToFloat64 => AstOperator::CastToFloat64,
        }
    }
}

/// Immutable owning cuDF AST expression tree.
///
/// Literal scalars are kept alive by this wrapper because cuDF AST literal
/// nodes reference scalar objects owned outside the tree.
///
/// Build expressions with [`CuDFAstExpressionBuilder`]. Finished expressions
/// are cheap to clone, so they can be retained while stream-ordered work is
/// still pending.
///
/// # Examples
///
/// ```no_run
/// use libcudf_rs::{CuDFAstExpression, CuDFAstOperator, CuDFAstTableReference};
///
/// # fn build() -> Result<(), libcudf_rs::CuDFError> {
/// let mut ast = CuDFAstExpression::builder();
/// let left = ast.column_reference(0, CuDFAstTableReference::Left)?;
/// let right = ast.column_reference(0, CuDFAstTableReference::Right)?;
/// ast.binary_operation(CuDFAstOperator::Equal, left, right)?;
/// let ast = ast.finish();
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct CuDFAstExpression {
    inner: Arc<CuDFAstExpressionInner>,
}

pub(crate) struct CuDFAstExpressionInner {
    inner: UniquePtr<ffi::AstExpressionTree>,
    literals: Vec<CuDFScalar>,
}

impl fmt::Debug for CuDFAstExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CuDFAstExpression").finish_non_exhaustive()
    }
}

impl CuDFAstExpression {
    /// Create a mutable builder for a cuDF AST expression tree.
    pub fn builder() -> CuDFAstExpressionBuilder {
        CuDFAstExpressionBuilder::new()
    }

    pub(crate) fn inner(&self) -> &UniquePtr<ffi::AstExpressionTree> {
        &self.inner.inner
    }
}

/// Mutable builder for a cuDF AST expression tree.
///
/// Each builder method appends a node and returns a [`CuDFAstNode`] that can be
/// used as input to later nodes. cuDF treats the most recently added node as
/// the expression root.
pub struct CuDFAstExpressionBuilder {
    inner: CuDFAstExpressionInner,
}

impl fmt::Debug for CuDFAstExpressionBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CuDFAstExpressionBuilder")
            .finish_non_exhaustive()
    }
}

impl CuDFAstExpressionBuilder {
    fn new() -> Self {
        Self {
            inner: CuDFAstExpressionInner {
                inner: ffi::ast_expression_tree_create(),
                literals: Vec::new(),
            },
        }
    }

    /// Finish this builder and return an immutable expression.
    pub fn finish(self) -> CuDFAstExpression {
        CuDFAstExpression {
            inner: Arc::new(self.inner),
        }
    }

    /// Add a column reference node.
    ///
    /// `column_index` is relative to the table selected by `table`. For join
    /// filters this is usually a projected conditional table, not necessarily
    /// the full input table.
    ///
    /// # Errors
    ///
    /// Returns an error if `column_index` cannot be represented as a cuDF
    /// column index or cuDF cannot add the column reference.
    pub fn column_reference(
        &mut self,
        column_index: usize,
        table: CuDFAstTableReference,
    ) -> Result<CuDFAstNode, CuDFError> {
        Ok(CuDFAstNode::from_raw(
            ffi::ast_expression_tree_add_column_reference(
                self.inner.inner.pin_mut(),
                column_index.try_into().map_err(|_| {
                    CuDFError::ArrowError(ArrowError::InvalidArgumentError(format!(
                        "AST column index {column_index} exceeds i32"
                    )))
                })?,
                table.into_sys() as i32,
            )?,
        ))
    }

    /// Add a column-name reference node.
    ///
    /// cuDF Parquet filters use names so filter columns do not need to be
    /// present in the projected output.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDF cannot add the column-name reference.
    pub fn column_name_reference(
        &mut self,
        column_name: impl AsRef<str>,
    ) -> Result<CuDFAstNode, CuDFError> {
        Ok(CuDFAstNode::from_raw(
            ffi::ast_expression_tree_add_column_name_reference(
                self.inner.inner.pin_mut(),
                column_name.as_ref(),
            )?,
        ))
    }

    /// Add a literal node.
    ///
    /// The scalar is moved into this expression tree wrapper and kept alive for
    /// as long as the AST exists.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDF cannot add the scalar as an AST literal.
    pub fn literal(&mut self, scalar: CuDFScalar) -> Result<CuDFAstNode, CuDFError> {
        self.inner.literals.push(scalar);
        let scalar = self.inner.literals.last().ok_or_else(|| {
            ArrowError::InvalidArgumentError("literal scalar insertion failed".to_string())
        })?;
        Ok(CuDFAstNode::from_raw(ffi::ast_expression_tree_add_literal(
            self.inner.inner.pin_mut(),
            scalar.inner(),
        )?))
    }

    /// Add a unary operation node.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` does not refer to an existing AST node or
    /// cuDF rejects the requested operation.
    pub fn unary_operation(
        &mut self,
        op: CuDFAstOperator,
        input: CuDFAstNode,
    ) -> Result<CuDFAstNode, CuDFError> {
        Ok(CuDFAstNode::from_raw(
            ffi::ast_expression_tree_add_unary_operation(
                self.inner.inner.pin_mut(),
                op.into_sys() as i32,
                input.into_raw(),
            )?,
        ))
    }

    /// Add a binary operation node.
    ///
    /// # Errors
    ///
    /// Returns an error if either operand does not refer to an existing AST
    /// node or cuDF rejects the requested operation.
    pub fn binary_operation(
        &mut self,
        op: CuDFAstOperator,
        left: CuDFAstNode,
        right: CuDFAstNode,
    ) -> Result<CuDFAstNode, CuDFError> {
        Ok(CuDFAstNode::from_raw(
            ffi::ast_expression_tree_add_operation(
                self.inner.inner.pin_mut(),
                op.into_sys() as i32,
                left.into_raw(),
                right.into_raw(),
            )?,
        ))
    }
}

impl Default for CuDFAstExpressionBuilder {
    fn default() -> Self {
        Self::new()
    }
}
