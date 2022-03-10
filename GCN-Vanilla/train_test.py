#author - Samantha Mahendran
import time
from model import Model, DataCleaning
from corpus_level import BuildGraph, TrainModel
from segment import Set_Connection


def segment(train, test, entites, no_rel=None, no_rel_multiple=False, parallelize=False, no_of_cores=64, down_sample=None, down_sample_ratio=0.2,
            predictions_folder=None):
    """
    Segmentation of the training and testing data
    :param train: path to train data
    :param test: path to test data
    :param entites: list of entities that create the relations
    :param no_rel: name the label when entities that do not have relations in a sentence are considered
    :param no_rel_multiple: flag whether multiple labels are possibles for No-relation
    :param predictions_folder: path to predictions (output) folder
    :param parallelize: parallelize the segmentation
    :param no_of_cores: number of cores used for parallelization
    :return: segments of train and test data
    """
    if no_rel:
        seg_train = Set_Connection(CSV=False, dataset=train, rel_labels=entites, no_labels=no_rel,
                                   no_rel_multiple=no_rel_multiple, write_Entites=False,
                                   parallelize=parallelize, no_of_cores=no_of_cores,
                                   predictions_folder=predictions_folder, down_sample=down_sample, down_sample_ratio=down_sample_ratio).data_object
    else:
        print("Starting segmentation of train data")
        seg_train = Set_Connection(CSV=False, dataset=train, rel_labels=entites, write_Entites=False,
                                   parallelize=parallelize, no_of_cores=no_of_cores,
                                   predictions_folder=predictions_folder, down_sample=down_sample, down_sample_ratio=down_sample_ratio).data_object
    time.sleep(50)
    print(" 5 sec more")
    time.sleep(5)
    print("Starting segmentation of test data")
    seg_test = Set_Connection(CSV=False, dataset=test, rel_labels=entites, test=True, parallelize=True,
                              write_Entites=True, no_of_cores=64, predictions_folder=predictions_folder, down_sample=down_sample, down_sample_ratio=down_sample_ratio).data_object

    return seg_train, seg_test


def run_GCN_model(seg_train, seg_test, embedding_path, embedding_binary, gcn_model, window_size, segment, entity_masking, replace_entity_pair, write_predictions=False,
                  write_No_rel=False, initial_predictions=None, final_predictions=None, train_path=None, test_path=None):
    """
    Choose the GCB model to run

    :param embedding_binary:
    :param seg_train: train segments
    :param seg_test: test segments
    :param embedding_path: path to the word embeddings
    :param gcn_model: choose the model
    :param window_size: size of the window applied over the words
    :type write_Predictions: write entities and predictions to file
    :param initial_predictions: folder to save the initial relation predictions
    :param final_predictions: folder to save the final relation predictions
    :param write_No_rel: Write the no-relation predictions back to files
    :return: None
    """
    if gcn_model == 'corpus-level':
        model = Model(data_object=seg_train, data_object_test=seg_test, segment=segment, entity_masking= entity_masking, replace_entity_pair = replace_entity_pair, test=True, train_path = train_path, test_path = test_path)
        clean_data = DataCleaning(model)
        graph = BuildGraph(model, clean_data, window_size, embedding_binary, word_embeddings_path=embedding_path)
        TrainModel(graph, write_Predictions=write_predictions, initial_predictions=initial_predictions,
                   final_predictions=final_predictions, write_No_rel=write_No_rel)

#         embedding = Embeddings(embedding_path, model, embedding_dim=embed_dim)
