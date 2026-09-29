---
title: 'Shape grouping'
linkTitle: 'Shape grouping'
weight: 11
description: 'Grouping multiple shapes during annotation.'
aliases:
  - /docs/manual/advanced/shape-grouping/
  - /docs/annotation/tools/shape-grouping/
---

This feature allows us to group several shapes.

You may use the `Group Shapes` button or shortcuts:

- `Alt+G` — start selection / end selection in group mode
- `Esc` — close group mode
- `Ctrl+Shift+G` — reset group for selected shapes

You may select shapes clicking on them or selecting an area.

Grouped shapes will have `group_id` filed in dumped annotation.

Also you may switch color distribution from an instance (default) to a group.
You have to switch `Color By Group` checkbox for that.

Shapes that don't have `group_id`, will be highlighted in white.

![Example of an annotation with grouped shapes](/images/image078_detrac.jpg)

![Example of an annotation with grouped and non-grouped shapes](/images/image077_detrac.jpg)
