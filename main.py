from xml_parse import xml_parse
from preprocess import normalize_corpus
from utils import build_feature_matrix
import document_clustering as dc
import pandas as pd


if __name__ == '__main__':
    # where corpus, titles, records_ids references array whereindex is 1:1 mapping for each records
    # where each data point is needed, use df which houses all data within a pandas dataframe
    corpus, titles, record_ids, df = xml_parse()
    nc = normalize_corpus(corpus)
    vectorizer, feature_matrix = build_feature_matrix(nc,
                                                      feature_type='tfidf',
                                                      min_df=0.24, max_df=0.85,
                                                      ngram_range=(1, 2))
    # get feature names
    feature_names = vectorizer.get_feature_names()
    ################################################################

    # specifically kmeans clustering of documents
    # num_clusters = 2
    # dc.find_optimal_cluster_num(feature_matrix, save_figure=True, max_n=25)
    # km_obj, clusters = dc.k_means(feature_matrix=feature_matrix, num_clusters=num_clusters)

    # # add normalized corpus as values to corpus series
    #
    # df['corpus'] = nc
    #
    # # add clusters to df DataFrame
    # df['Cluster'] = clusters
    #
    # # # get clustering analysis data
    # cluster_data = dc.get_cluster_data(clustering_obj=km_obj, data=df,feature_names=feature_names,
    #                                    num_clusters=num_clusters, topn_features=3)
    #
    # # visualize clusters
    # dc.plot_clusters(num_clusters=num_clusters, feature_matrix=feature_matrix, cluster_data=cluster_data,
    #                  data=df, plot_size=(16, 8), save_figure=True)
    #
    # dc.print_cluster_data(cluster_data)
    # ################################################################
    # Wards Hierarchical Clustering
    linkage_matrix = dc.ward_hierarchical_clustering(feature_matrix)
    # plot dendogram
    dc.plot_hierarchical_clusters(linkage_matrix=linkage_matrix, data=df, figure_size=(8,12))



