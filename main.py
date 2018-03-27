import sys
import util
import graph_util as gu
from graph import CDNK_Vectorizer


if __name__=='__main__':
    #if len(sys.argv) < 2:
    #    sys.exit("python main_netkit.py dataset iterations")
        
    #adjacency_folder = sys.argv[1]
    #deg_threshold = int(sys.argv[2])  
    #cli_threshold = int(sys.argv[3])
    #max_node_ID = int(sys.argv[4])

    #r = int(sys.argv[5])
    #d = int(sys.argv[6])

    adjacency_folder = "/media/dinh/DATA/Test_ECCB/adjs/"
    deg_threshold = 10 
    cli_threshold = 4
    max_node_ID = 2445
    r = 2
    d = 3
    
    graphs = gu.create_graphs(adjacency_folder_path= adjacency_folder)
    print "Done creating graphs"
    
    g_union = gu.union_graphs(graphs=graphs, deg_threshold=deg_threshold, cli_threshold=cli_threshold, max_node_ID=max_node_ID)  
    print "Done union"
    
    vec = CDNK_Vectorizer(r=2, d=2)    
    M = vec.vectorize(g_union)
    print M.shape
    
    K = vec.cdnk(g_union)    
    print K.shape
    
    print "Done"