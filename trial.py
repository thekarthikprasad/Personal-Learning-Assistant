# 
# Trial to check and test with functions defined in the application-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




graph_q = {'entities': [{'label': 'concept', 'id': 'concept1', 'conceptname': 'Hadoop'}, {'label': 'concept', 'id': 'concept2', 'conceptname': 'Data'}, {'label': 'concept', 'id': 'concept3', 'conceptname': 'Digital Universe'}, {'label': 'concept', 'id': 'concept4', 'conceptname': 'Data Storage and Analysis'}, {'label': 'concept', 'id': 'concept5', 'conceptname': 'MapReduce'}, {'label': 'concept', 'id': 'concept6', 'conceptname': 'Batch Processing'}, {'label': 'concept', 'id': 'concept7', 'conceptname': 'Distributed Computing'}, {'label': 'concept', 'id': 'concept8', 'conceptname': 'Data Processing Patterns'}], 'relationships': ['concept1 | relatedto | concept2', 'concept2 | relatedto | concept3', 'concept2 | relatedto | concept4', 'concept4 | relatedto | concept5', 'concept5 | relatedto | concept6', 'concept7 | relatedto | concept8', 'concept1 | usedfor | concept4', 'concept5 | usedfor | concept4']}

for entity in graph_q['entities']:
    print(entity['conceptname'])


