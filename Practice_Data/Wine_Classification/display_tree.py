from sklearn.externals.six import StringIO
from sklearn import tree
import pydotplus as pydot

def save_to_pdf(clf, feature_names, label_names, filename):
    dot_data = StringIO()
    tree.export_graphviz(clf,
                         out_file = dot_data,
                         feature_names = feature_names,
                         class_names = label_names,
                         filled = True, rounded = True,
                         impurity = False)
    
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(filename+'.pdf')





