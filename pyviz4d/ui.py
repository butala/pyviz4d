import vtk

def create_slider(iren, title, min_val, max_val, initial_val, pos_y, callback):
    """
    Creates a 2D slider widget in the VTK window.
    pos_y should be between 0.0 and 1.0 (normalized viewport coordinate).
    """
    slider_rep = vtk.vtkSliderRepresentation2D()
    slider_rep.SetMinimumValue(min_val)
    slider_rep.SetMaximumValue(max_val)
    slider_rep.SetValue(initial_val)
    slider_rep.SetTitleText(title)

    # Position: Normalized Viewport coordinates
    slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedViewport()
    slider_rep.GetPoint1Coordinate().SetValue(0.75, pos_y)
    slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedViewport()
    slider_rep.GetPoint2Coordinate().SetValue(0.95, pos_y)

    slider_rep.SetSliderLength(0.02)
    slider_rep.SetSliderWidth(0.03)
    slider_rep.SetEndCapLength(0.01)
    slider_rep.SetEndCapWidth(0.03)
    slider_rep.SetTubeWidth(0.005)
    slider_rep.SetLabelFormat("%g")

    slider_widget = vtk.vtkSliderWidget()
    slider_widget.SetInteractor(iren)
    slider_widget.SetRepresentation(slider_rep)
    slider_widget.SetAnimationModeToAnimate()
    slider_widget.EnabledOn()

    slider_widget.AddObserver("InteractionEvent", callback)

    return slider_widget

def create_text_actor(text, pos_x, pos_y, font_size=16):
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(text)
    text_property = text_actor.GetTextProperty()
    text_property.SetFontSize(font_size)
    text_property.SetColor(1.0, 1.0, 1.0)

    coord = text_actor.GetPositionCoordinate()
    coord.SetCoordinateSystemToNormalizedViewport()
    coord.SetValue(pos_x, pos_y)

    return text_actor
