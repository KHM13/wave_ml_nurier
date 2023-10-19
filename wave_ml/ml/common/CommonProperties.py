import os
import configparser
from configparser import SectionProxy
from wave_ml.config.settings.base import BASE_DIR


class CommonProperties:
    server: SectionProxy = {}
    spark: SectionProxy = {}
    file_path: SectionProxy = {}
    file_name: SectionProxy = {}
    data: SectionProxy = {}
    preprocess: SectionProxy = {}
    outlier: SectionProxy = {}
    feature_select: SectionProxy = {}
    model_training: SectionProxy = {}
    model: SectionProxy = {}
    prediction: SectionProxy = {}
    elasticsearch: SectionProxy = {}
    listener: SectionProxy = {}
    redis: SectionProxy = {}
    scheduler: SectionProxy = {}
    isServerRunning = False
    isOperationServer = False

    spark_app_name = "nurier_fds_ml"
    type_name_IB = "model_IB"
    type_name_SB = "model_SB"

    def __init__(self):
        properties = configparser.ConfigParser()
        properties.read(f'{BASE_DIR}\\ml\\resource\\config.ini', encoding='UTF-8')

        self.server = properties['SERVER']
        self.spark = properties['SPARK']
        self.file_path = properties['FILEPATH']
        self.file_name = properties['FILENAME']
        self.data = properties['DATA']
        self.preprocess = properties['PREPROCESS']
        self.outlier = properties['OUTLIER']
        self.feature_select = properties['FEATURESELECT']
        self.model_training = properties['MODELTRAINING']
        self.model = properties['MODEL']
        self.prediction = properties['PREDICTION']
        self.elasticsearch = properties['ELASTICSEARCH']
        self.listener = properties['LISTENER']
        self.sender = properties['SENDER']
        self.redis = properties['REDIS']
        self.scheduler = properties['SCHEDULER']

        self.isOperationServer = True if self.server.get('server.type') == 'operation' else False

    @staticmethod
    def is_server() -> bool:
        os_name = os.name
        if os_name.__contains__("nt"):  # nt : 윈도우 / posix : 리눅스
            return False
        else:
            return True

    def get_server_type(self) -> str:
        return self.server.get('server.type')

    def get_key(self, key) -> str:
        if self.isOperationServer:
            return "operation." + key
        else:
            return "development." + key

    def get_key2(self, key) -> str:
        if self.isOperationServer:
            return "operation." + key
        else:
            if self.is_server():
                return "development." + key
            else:
                return "windows." + key

    def get_spark_user_dir(self) -> str:
        return self.spark.get(self.get_key2('spark.user.dir'))

    def get_spark_executor_cores(self) -> str:
        return self.spark.get('spark.executor.cores')

    def get_spark_master(self) -> str:
        return self.spark.get("spark.master")

    def get_preprocess_file_path(self) -> str:
        return self.file_path.get(self.get_key2('preprocess.result.file.path'))

    def get_result_log_file_path(self) -> str:
        return self.file_path.get(self.get_key2('result.log.file.path'))

    def get_model_report_file_path(self) -> str:
        return self.file_path.get(self.get_key2('model.report.file.path'))

    def get_model_result_file_path(self) -> str:
        return self.file_path.get(self.get_key2('model.result.file.path'))

    def get_original_data_file_path(self) -> str:
        return self.file_path.get(self.get_key2('original.data.file.path'))

    def get_preprocess_result_file_name(self) -> str:
        return self.file_name.get('preprocess.result.file.name')

    def get_preprocess_log_file_name(self) -> str:
        return self.file_name.get('preprocess.log.file.name')

    def get_outlier_log_file_name(self) -> str:
        return self.file_name.get('outlier.log.file.name')

    def get_feature_select_log_file_name(self) -> str:
        return self.file_name.get('feature.select.log.file.name')

    def get_original_data_file_name(self) -> str:
        return self.file_name.get('original.data.file.name')

    def get_message_ib_data_file_name(self) -> str:
        return self.file_name.get('message.ib.data.file.name')

    def get_message_sb_data_file_name(self) -> str:
        return self.file_name.get('message.sb.data.file.name')

    def get_prediction_ib_training_data(self) -> str:
        return self.data.get(self.get_key2('model.prediction.IB.model.trainingData'))

    def get_prediction_sb_training_data(self) -> str:
        return self.data.get(self.get_key2('model.prediction.SB.model.trainingData'))

    def get_preprocess_media_type_group(self) -> list:
        return self.preprocess.get('preprocess.mediaType.group').split(";")

    def get_preprocess_execute_scaler(self) -> str:
        return self.preprocess.get('preprocess.executeScaler')

    def get_outlier_min_standard(self) -> float:
        return self.outlier.getfloat('outlier.min.standard')

    def get_outlier_max_standard(self) -> float:
        return self.outlier.getfloat('outlier.max.standard')

    def get_outlier_rate(self) -> int:
        return self.outlier.getint('outlier.rate')

    def get_feature_select_column_ib_list(self) -> str:
        return self.feature_select.get('feature.select.column.IB.list')

    def get_feature_select_column_sb_list(self) -> str:
        return self.feature_select.get('feature.select.column.SB.list')

    def get_model_training_execute_model_list(self) -> str:
        return self.model_training.get('modelTraining.executeModel.list')

    def get_model_training_drop_column_list(self) -> str:
        return self.model_training.get('modelTraining.drop.Column.list')

    def get_model_training_drop_column_ib_list(self) -> str:
        return self.model_training.get('modelTraining.drop.Column.IB.list')

    def get_model_training_drop_column_sb_list(self) -> str:
        return self.model_training.get('modelTraining.drop.Column.SB.list')

    def is_model_report_save(self) -> bool:
        return self.model.getboolean('model.report.isSave')

    def is_model_result_save(self) -> bool:
        return self.model.getboolean('model.result.isSave')

    def get_model_max_beans(self) -> int:
        return self.model.getint('model.max.beans')

    def get_model_training_size(self) -> float:
        return self.model.getfloat('model.train.size')

    def get_prediction_ib_is_load(self) -> bool:
        return self.model.getboolean('model.prediction.IB.isLoad')

    def get_prediction_ib_model_name(self) -> str:
        return self.prediction.get('model.prediction.IB.model.name')

    def get_prediction_ib_model_param_maxiter(self) -> int:
        return self.prediction.getint('model.prediction.IB.model.param.maxIter')

    def get_prediction_ib_model_param_reg(self) -> float:
        return self.prediction.getfloat('model.prediction.IB.model.param.reg')

    def get_prediction_ib_model_param_fit_intercept(self) -> bool:
        return self.prediction.getboolean('model.prediction.IB.model.param.fitIntercept')

    def get_prediction_ib_model_param_elasticnet(self) -> float:
        return self.prediction.getfloat('model.prediction.IB.model.param.elasticNet')

    def get_prediction_ib_model_param_maxdepth(self) -> int:
        return self.prediction.getint('model.prediction.IB.model.param.maxDepth')

    def get_prediction_ib_model_param_layer_options(self) -> int:
        return self.prediction.getint('model.prediction.IB.model.param.layerOptions')

    def get_prediction_ib_model_param_block_size(self) -> int:
        return self.prediction.getint('model.prediction.IB.model.param.blockSize')

    def get_prediction_ib_model_execute_scaler(self) -> str:
        return self.prediction.get('model.prediction.IB.model.executeScaler')

    def get_prediction_ib_model_outlier_used(self) -> bool:
        return self.prediction.getboolean('model.prediction.IB.model.outlier.used')

    def get_prediction_ib_model_media_type_group(self) -> str:
        return self.prediction.get('model.prediction.IB.model.mediaType.group')

    def get_prediction_sb_is_load(self) -> bool:
        return self.model.getboolean('model.prediction.SB.isLoad')

    def get_prediction_sb_model_name(self) -> str:
        return self.prediction.get('model.prediction.SB.model.name')

    def get_prediction_sb_model_param_maxiter(self) -> int:
        return self.prediction.getint('model.prediction.SB.model.param.maxIter')

    def get_prediction_sb_model_param_reg(self) -> float:
        return self.prediction.getfloat('model.prediction.SB.model.param.reg')

    def get_prediction_sb_model_param_fit_intercept(self) -> bool:
        return self.prediction.getboolean('model.prediction.SB.model.param.fitIntercept')

    def get_prediction_sb_model_param_elasticnet(self) -> float:
        return self.prediction.getfloat('model.prediction.SB.model.param.elasticNet')

    def get_prediction_sb_model_param_maxdepth(self) -> int:
        return self.prediction.getint('model.prediction.SB.model.param.maxDepth')

    def get_prediction_sb_model_param_layer_options(self) -> int:
        return self.prediction.getint('model.prediction.SB.model.param.layerOptions')

    def get_prediction_sb_model_param_block_size(self) -> int:
        return self.prediction.getint('model.prediction.SB.model.param.blockSize')

    def get_prediction_sb_model_execute_scaler(self) -> str:
        return self.prediction.get('model.prediction.SB.model.executeScaler')

    def get_prediction_sb_model_outlier_used(self) -> bool:
        return self.prediction.getboolean('model.prediction.SB.model.outlier.used')

    def get_prediction_sb_model_media_type_group(self) -> str:
        return self.prediction.get('model.prediction.SB.model.mediaType.group')

    def get_search_engine_ml_server_nodes(self) -> str:
        return self.elasticsearch.get(self.get_key('searchEngine.ml.server.nodes'))

    def get_search_engine_ml_server_cluster_name(self) -> str:
        return self.elasticsearch.get(self.get_key('searchEngine.ml.server.clusterName'))

    def get_search_engine_ml_server_user_name(self) -> str:
        return self.elasticsearch.get(self.get_key('searchEngine.ml.server.userName'))

    def get_search_engine_ml_server_user_password(self) -> str:
        return self.elasticsearch.get(self.get_key('searchEngine.ml.server.userPassword'))

    def get_search_engine_cluster_name(self) -> str:
        return self.elasticsearch.get('searchEngine.cluster.name')

    def get_search_engine_index_name(self) -> str:
        return self.elasticsearch.get('searchEngine.index.name')

    def get_search_engine_index_type_message(self) -> str:
        return self.elasticsearch.get('searchEngine.index.type.Message')

    def get_search_engine_ml_index_name(self) -> str:
        return self.elasticsearch.get('searchEngine.ml.index.name')

    def get_search_engine_ml_type_message(self) -> str:
        return self.elasticsearch.get('searchEngine.ml.type.Message')

    def get_listener_engine_kafka_ml_topic_name(self) -> str:
        return self.listener.get('listenerEngine.kafka.ml.topic.name')

    def get_listener_engine_kafka_ml_group_id(self) -> str:
        return self.listener.get('listenerEngine.kafka.ml.group.id')

    def get_listener_engine_ml_servers(self) -> list:
        servers = self.listener.get(self.get_key('listenerEngine.ml.servers'))
        server_list: list = servers.split(",") if servers.__contains__(",") else [servers]
        return server_list

    def get_sender_engine_kafka_ml_topic_name(self) -> str:
        return self.sender.get('senderEngine.kafka.ml.topic.name')

    def get_sender_engine_kafka_ml_group_id(self) -> str:
        return self.sender.get('senderEngine.kafka.ml.group.id')

    def get_sender_engine_ml_servers(self) -> list:
        servers = self.sender.get(self.get_key('senderEngine.ml.servers'))
        server_list: list = servers.split(",") if servers.__contains__(",") else [servers]
        return server_list

    def get_save_engine_ml_servers(self) -> list:
        servers = self.redis.get(self.get_key('saveEngine.redis.ml.servers'))
        server_list: list = servers.split(";") if servers.__contains__(";") else [servers]
        return server_list

    def get_ml_storage_sender_fix_thread_count(self) -> int:
        return self.scheduler.getint('worker.MLStorageSender.FixThreadCount')

    def get_kafka_sender_fix_thread_count(self) -> int:
        return self.scheduler.getint('worker.KafkaSender.FixThreadCount')

    def get_update_prediction_fix_thread_count(self) -> int:
        return self.scheduler.getint('worker.UpdatePrediction.FixThreadCount')

    def get_prediction_schedule_period_ms(self) -> int:
        return self.scheduler.getint('worker.FDSPredictionSchedule.period.ms')

    def get_prediction_schedule_datarow_max(self) -> int:
        return self.scheduler.getint('worker.FDSPredictionSchedule.dataRow.max')

    def set_spark_app_name(self, app_name):
        self.spark_app_name = app_name

    def get_spark_app_name(self):
        return self.spark_app_name

    def set_server_running(self, is_server_running):
        self.isServerRunning = is_server_running
