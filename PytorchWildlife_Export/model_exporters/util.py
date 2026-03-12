import onnx

def merge_onnx_models(m_1, m_2, prefix1, prefix2):
    if isinstance(m_1, str):
        m_1 = onnx.load(m_1)
    if isinstance(m_2, str):
        m_2 = onnx.load(m_2)

    # assumes that we only have one of these inputs/outputs! should work ok for YOLO
    m_1_output_name = [node.name for node in m_1.graph.output][0]
    m_2_input_name = [node.name for node in m_2.graph.input][0]

    merged_model = onnx.compose.merge_models(
        m_1,
        m_2,
        io_map=[(m_1_output_name, m_2_input_name)],
        prefix1=prefix1,
        prefix2=prefix2,
    )
    return merged_model