//! Diagnostics.

use codespan::{FileId, Span};
use codespan_reporting::diagnostic::{Diagnostic, Label, Severity};

use crate::stratified;

pub mod warning {
    use super::*;

    pub fn host_type_found_in_field(
        file_id: FileId,
        struct_span: Span,
        host_ty_span: Span,
    ) -> Diagnostic {
        Diagnostic {
            severity: Severity::Warning,
            code: None,
            message: "host type encountered in `struct` field".to_owned(),
            primary_label: Label::new(file_id, host_ty_span, "host type"),
            secondary_labels: vec![Label::new(
                file_id,
                struct_span,
                "`struct` contains non-format types",
            )],
            notes: vec![
                "only format types can appear in `struct` fields".to_owned(),
                "this `struct` will not appear in the compiled parser".to_owned(),
            ],
        }
    }
}

pub mod error {
    use super::*;

    pub fn unconstrained_int(file_id: FileId, span: Span) -> Diagnostic {
        Diagnostic {
            severity: Severity::Error,
            code: None,
            message: "cannot compile unconstrained integer types".to_owned(),
            primary_label: Label::new(file_id, span, "unconstrained integer type"),
            secondary_labels: vec![],
            notes: vec![],
        }
    }
}

pub mod bug {
    pub use super::*;

    pub fn item_name_reused(
        file_id: FileId,
        name: &stratified::Label,
        found: Span,
        original: Span,
    ) -> Diagnostic {
        Diagnostic {
            severity: Severity::Bug,
            code: None,
            message: format!(
                "attempted to compile an item named `{}` multiple times",
                name,
            ),
            primary_label: Label::new(file_id, found, "redefined here"),
            secondary_labels: vec![Label::new(file_id, original, "original item here")],
            notes: vec![format!(
                "`{}` must be defined only once in this module",
                name,
            )],
        }
    }

    pub fn unbound_item(file_id: FileId, name: &stratified::Label, span: Span) -> Diagnostic {
        Diagnostic {
            severity: Severity::Bug,
            code: None,
            message: format!(
                "attempted to compile an item `{}` that was not yet bound",
                name,
            ),
            primary_label: Label::new(file_id, span, "item not found in this scope"),
            secondary_labels: vec![],
            // TODO: provide suggestions
            notes: vec![],
        }
    }

    pub fn non_format_type_as_host_type(file_id: FileId, span: Span) -> Diagnostic {
        Diagnostic {
            severity: Severity::Bug,
            code: None,
            message: "attempted to compile a non-format type as a host type".to_owned(),
            primary_label: Label::new(file_id, span, "not a format type"),
            secondary_labels: vec![],
            notes: vec![],
        }
    }
}
