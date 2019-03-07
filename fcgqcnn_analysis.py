from gqcnn.analysis import GQCNNAnalyzer
from autolab_core import YamlConfig

analysis_config = YamlConfig('cfg/tools/analyze_gqcnn_performance.yaml')
analyzer = GQCNNAnalyzer(analysis_config)
analyzer.analyze('models/FC-GQCNN-4.0-PJ','/home/amithp/fcgqcnn_env/output')

