# NotebookRead Cell ID Issue

## Error Description
When using `NotebookRead` to read the entire notebook, cells display with IDs like "cell-0", "cell-1", etc. However, when attempting to read individual cells using these exact same IDs with the `cell_id` parameter, the tool returns "Cell with ID 'cell-1' not found in notebook" errors.

## Reproduction Steps
1. Use `NotebookRead` without `cell_id` parameter - successfully shows cells with IDs "cell-0", "cell-1", etc.
2. Use `NotebookRead` with `cell_id="cell-1"` - fails with "Cell with ID 'cell-1' not found"
3. Tried variations like `cell_id="0"` and `cell_id=0` - all fail

## Current Workaround
Always read the full notebook structure first rather than trying to read individual cells. The `NotebookEdit` tool appears to work correctly with these cell IDs, suggesting the issue is specific to `NotebookRead` when using the `cell_id` parameter.

## Impact
This makes it impossible to read individual cells for inspection, requiring full notebook reads even when only examining specific cells.