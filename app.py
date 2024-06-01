import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import time
import re
from string import Template
import json
import pandas as pd
import getpass
from neo4j import GraphDatabase
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import UnstructuredPDFLoader

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain.agents import load_tools
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
import os
from utils import read_pdf
from dotenv import load_dotenv,dotenv_values
from langchain_text_splitters import NLTKTextSplitter
import nltk

load_dotenv()
llm_calls = 0
groq_key_m = os.environ['GROQ_API_KEY_M']
groq_key_l = os.environ['GROQ_API_KEY_L']
url_neo =os.environ['NEO4J_URL']
url_pw = os.environ['NEO4J_PASS']


llm = ChatGroq(temperature=0, groq_api_key=groq_key_m, model_name="mixtral-8x7b-32768")

st.title("Assist")
prompt_concepts = """
From the text book, extract entities strictly as below
1. First go through the text thoroughly
2. Next you should semantically understand what the piece of text is about
3. Now that you understood the context, carefully read through the text and find the underlying concepts
4. extract the key concepts with respect to the text.
EXTRACT ONLY KEY CONCEPTS
now extract the needed info below
 'id' property of each entity must be alphanumeric+hex+conceptname's 2 letters which must be unique among the entities
 you will be referring this property to define the relationship between the entities.
 NEVER create new entity types that are not mentioned below. Now the extracted concept must be named (the naming of the concept must refer to
 the text, and name accordingly)
 and stored inside concept entity under 'concept_name' property.

If you dont find any info of entities and relationships its okay to return empty value. Do NOT create fictious data

5. Do not create duplicate entities
6. restrict yourself to extract only key concept from text. No other unnecessary things to be focused
7. Model relationships in a way that the concepts are related hierarchically

Example output JSON:

{{
  "entities":[

  {{"label":"concept","id":"concept1,"conceptname":"name of the identified concept"}},

{{"label":"concept","id":"concept2,"conceptname": "name of the identified concept"
}}
],

"relationships":
["concept1 | relatedto | concept2", "concept3 | relatedto | concept4" ]
}}

Don't output unnecessary things, the output contain the critical key concepts and should be very concise

Question: Extract the concept, using this text book text: {etext}

{format_instructions}
Answer:
"""


# prompt = PromptTemplate(
#     template=prompt_concepts,
#     input_variables=["etext"],
#     partial_variables={"format_instructions": parser.get_format_instructions()},
# )


# File Upload

uploaded_file = st.file_uploader("Choose a file", type=['pdf'])

# Display file details if a file is uploaded
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    st.write(file_details)

    # Read and display file content
    file_content = read_pdf.extract_text_from_pdf(uploaded_file)
    #st.write(file_content)
    #st.write("File Content:")
    #st.write(file_content)
#file_content = extract_text_from_pdf(uploaded_file)



# Clean KG

def get_prop_str(prop_dict, _id):
    s = []
    for key, val in prop_dict.items():
      if key != 'label' and key != 'id':
         s.append(_id+"."+key+' = "'+str(val).replace('\"', '"').replace('"', '\"')+'"')
    return ' ON CREATE SET ' + ','.join(s)

def get_cypher_compliant_var(_id):
    s = "_"+ re.sub(r'[\W_]', '', _id).lower() #avoid numbers appearing as firstchar; replace spaces
    return s[:20] #restrict variable size

def generate_cypher(file_name, in_json):
    e_map = {}
    e_stmt = []
    r_stmt = []
    e_stmt_tpl = Template("($id:$label{id:'$key'})")
    r_stmt_tpl = Template("""
      MATCH $src
      MATCH $tgt
      MERGE ($src_id)-[:$rel]->($tgt_id)
    """)
    for obj in in_json:
      for j in obj['entities']:
          props = ''
          label = j['label']
          id = ''
          if label == 'conceptname':
            id = 'p'+str(file_name)
          # elif label == 'Position':
          #   c = j['id'].replace('position', '_')
          #   id = f'j{str(file_name)}{c}'
          # elif label == 'Education':
          #   c = j['id'].replace('education', '_')
          #   id = f'e{str(file_name)}{c}'
          else:
            id = get_cypher_compliant_var(j['conceptname'])
          if label in ['concept']:
            varname = get_cypher_compliant_var(j['id'])
            stmt = e_stmt_tpl.substitute(id=varname, label=label, key=id)
            e_map[varname] = stmt
            e_stmt.append('MERGE '+ stmt + get_prop_str(j, varname))

      for st in obj['relationships']:
          rels = st.split("|")
          src_id = get_cypher_compliant_var(rels[0].strip())
          rel = rels[1].strip()
          if rel in ['related_to','type']: #we ignore other relationships
            tgt_id = get_cypher_compliant_var(rels[2].strip())
            stmt = r_stmt_tpl.substitute(
              src_id=src_id, tgt_id=tgt_id, src=e_map[src_id], tgt=e_map[tgt_id], rel=rel)
            r_stmt.append(stmt)

    return e_stmt, r_stmt


import json

def convert_to_json(input_str):
    try:
        # Remove escape sequences (\n)
        cleaned_input_str = input_str.replace("\\n", "")
        
        # Parse as JSON
        json_data = json.loads(cleaned_input_str)
        return json_data
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None



def clean_text(text):
    import re
    return re.sub(r'[^\x00-\x7F]+',' ', text)

import networkx as nx
import matplotlib.pyplot as plt
G = nx.DiGraph()

if 'concept_index' not in st.session_state:
    st.session_state.concept_index = 0
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'retriever' not in st.session_state:
    st.session_state.retriever = None




identified_concepts = []
relations = []

import time
memory = ConversationBufferMemory()
from pydantic import BaseModel, Field
from typing import List

knowledge_graphs = {
    'entities':[],
    'relationships':[]
}
def merge_2_jsons(json1,json2):
    """merge 2 jsons with attributes - entitites, relationships"""
    
    json2['entities']+=json1['entities']
    json2['relationships']+=json1['relationships']
    return json2






if st.button('Generate Knowledge Graph'):

    import time
    memory = ConversationBufferMemory()
    from pydantic import BaseModel, Field
    from typing import List


    parser = JsonOutputParser()
    prompt_new = PromptTemplate.from_template(template=prompt_concepts,
    partial_variables={"format_instructions": parser.get_format_instructions()})
    chain = LLMChain(
            llm=llm,
            prompt=prompt_new,
            verbose=True,
            memory=memory)
    chain = prompt_new | llm | parser
    concep = chain.invoke({"etext":file_content })
    print(concep)
    #print(concepts_json)
    st.write("before going in concep")
    st.write(concep)
    time.sleep(60)




    ent_cyp, rel_cyp = generate_cypher('my_kg', [concep])
    print(ent_cyp, rel_cyp)

    st.write(concep)

    print(concep)





    # Knowledge Graph Generation
    def run_query(query, params={}):
        with driver.session() as session:
            result = session.run(query, params)
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    connectionUrl = "neo4j+s://3d5cc735.databases.neo4j.io"
    username ="neo4j"
    password = "SXy9XNhzdA3ZJHsgJfqZaZDpztO_n_OBqYIGtosMv00"

    driver = GraphDatabase.driver(connectionUrl, auth=(username, password))
    driver.verify_connectivity()

    import pandas as pd

    for e in ent_cyp:
        run_query(e)

    for r in rel_cyp:
        run_query(r)


    for entity in concep['entities']:
        G.add_node(entity['id'], label=entity['conceptname'])

    for relationship in concep['relationships']:
        source, _, target = relationship.split(' | ')
        G.add_edge(source, target)

    pos = nx.spring_layout(G)
    fig = nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold", arrowsize=20)
    plt.title("Concept Graph")
    st.pyplot(fig)

    for i in concep['entities']:
        concept = i['conceptname']
        identified_concepts.append(concept)
    st.write(identified_concepts)

    for relationship in concep['relationships']:
        source, relation, target = relationship.split(' | ')
        relations.append(relation)
st.write(relations)

if 'quess' not in st.session_state:
    st.session_state.quess = None

if 'a' not in st.session_state:
    st.session_state.a = None

if 'concepts' not in st.session_state:
        st.session_state.concepts=0

if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

if 'concept_index' not in st.session_state:
    st.session_state.concept_index = 0


from langchain.document_loaders import PyPDFLoader

def initialize_rag():
    loader = PyPDFLoader("subset.pdf")
    pages_1 = loader.load()
    vectorstore = Chroma.from_documents(
        documents=pages_1,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()
    return retriever

# Initialize RAG retriever only once
if st.session_state.retriever is None:
    st.session_state.retriever = initialize_rag()

# Function to generate a question using RAG
def generate_question(concept):
    after_rag_template = """Ask me questions based on the context only
        {context}
        Question: {question}
        """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": st.session_state.retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | llm
        | StrOutputParser()
    )
    question = after_rag_chain.invoke(f"Ask me questions from this concept {concept}")
    return question

def get_concepts(file_content):
    text_splitter = NLTKTextSplitter(chunk_size=3000)
    texts = text_splitter.split_text(file_content)
    length = len(texts)
    wait = len(file_content)/3000
    st.write("your wait time is : ",wait)
    for text in texts:
        parser = JsonOutputParser()
        prompt_new = PromptTemplate.from_template(template=prompt_concepts,
        partial_variables={"format_instructions": parser.get_format_instructions()})
        chain = LLMChain(
            llm=llm,
            prompt=prompt_new,
            verbose=True,
            memory=memory)
    
        chain = prompt_new | llm | parser
        concep = chain.invoke({"etext": text})
        #llm_calls+=1
    # print(concep)
        #print("this your kg")
        #st.write("this your captured kg")
        kg = merge_2_jsons(concep,knowledge_graphs)
    #print(knowledge_graphs)

    #print(concep)
    #print(concepts_json)
        #st.write("before going in concep")
        for i in knowledge_graphs['entities']:
            identified_concepts.append(i['conceptname'])
        time.sleep(60)

        return identified_concepts

    #print(identified_concepts)
    #st.write(concep)
    #st.write(text)
        



# Button to generate the next question
if st.button("get-conc"):

    conceptss = get_concepts(file_content)

if st.button('Generate Question v1'):
    st.write("Generating question using RAG")
    #print(conceptss)
    current_concept = identified_concepts[st.session_state.concept_index]
    st.session_state.current_question = generate_question(current_concept)
    st.write(st.session_state.current_question)

# Display the current question and input for the answer
st.write(st.session_state.current_question)
st.session_state.a = st.text_input("Enter your answers", key="answer")

# Button to submit the answer
if st.button("Submit Answers"):
    st.session_state.answers.append(st.session_state.a)
    st.session_state.a = ""
    st.session_state.concept_index += 1
    
    if st.session_state.concept_index < len(identified_concepts):
        current_concept = identified_concepts[st.session_state.concept_index]
        st.session_state.current_question = generate_question(current_concept)
    else:
        st.session_state.current_question = None
    
    st.experimental_rerun()

# Display completion message and answers if all questions are done
if st.session_state.concept_index >= len(identified_concepts):
    st.write("You have completed all questions.")
    st.write("Your answers:", st.session_state.answers)












if st.button('Generate Question'):  
    st.write("this is goin to be rag")
    for text in texts:
        parser = JsonOutputParser()
        prompt_new = PromptTemplate.from_template(template=prompt_concepts,
        partial_variables={"format_instructions": parser.get_format_instructions()})
        chain = LLMChain(
            llm=llm,
            prompt=prompt_new,
            verbose=True,
            memory=memory)
    
        chain = prompt_new | llm | parser
        concep = chain.invoke({"etext": text})
        llm_calls+=1
    # print(concep)
        print("this your kg")
        st.write("this your captured kg")
        merge_2_jsons(concep,knowledge_graphs)
    #print(knowledge_graphs)
        st.write(knowledge_graphs)

    #print(concep)
    #print(concepts_json)
        st.write("before going in concep")
        for i in knowledge_graphs['entities']:
            identified_concepts.append(i['conceptname'])

    #print(identified_concepts)
    #st.write(concep)
    #st.write(text)
        time.sleep(60)

    current_concept = identified_concepts[st.session_state.concept_index]
    
    
    from langchain_community.document_loaders import TextLoader
    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader("subset.pdf")
    pages_1 = loader.load()

    

    vectorstore = Chroma.from_documents(
    documents=pages_1,
    collection_name="rag-chroma",
    embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()
    print("pre q state")

    
    after_rag_template = """Ask me questions based on the context only
        {context}
        Question: {question}
        """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
            | after_rag_prompt
            | llm
            | StrOutputParser()
            )
    
    st.write("We are in the loop")
            
    #st.session_state.q = i
    # st.session_state.quess = after_rag_chain.invoke(f"Ask me questions from this concept {identified_concepts[st.seesion_state.concepts]}")
    # st.write(st.session_state.quess)
    current_concept = identified_concepts[st.session_state.concept_index]
    st.session_state.current_question = after_rag_chain.invoke(f"Ask me questions from this concept {current_concept}")
    st.write(st.session_state.current_question)
    st.session_state.a = st.text_input("Enter your answers")
    st.write(st.session_state.a)

    if st.button("Submit Answers"):
                

                question = st.session_state.quess
                answer = st.session_state.a
                quessa = "questions:"+ question + "\n" "\n" "\n" "\n"+ "answers:"+ answer
                memory = ConversationBufferMemory()
                quessa = "questions:"+"\n"+ question + "\n" "\n" "\n" "\n"+ "user answers:"+ "\n" + answer
                st.write(quessa)

                brain = ChatGroq(temperature=0, groq_api_key=groq_key_l, model_name="llama3-70b-8192")
                evaluation_prompt = """
            You are an excellent evaluator who conceptually and semantically evaluate the answers of the user.
            you go through the questions and answers, check the correctness.
            carefully analyse whether the user is correct or not.
            on a scale of 10 evaluate every user answer.
            after evaluating aggregate the score for the questions.
            dont hallucinate

            finally output a single aggregate score - which represents user's skill level.


            eg: 
            questions:
            q1 - 5, q2- 5, q3-5

            output: 
            Aggregate score: 

            Expected Output:

            Aggregate: score
            Remarks: r1,r2,....
            use this qna data pair: {qna}

            """
                print("entered here")
                prompt = PromptTemplate.from_template(evaluation_prompt)
                chain = LLMChain(
                llm=brain,
                prompt=prompt,
                verbose=True,
                memory=memory)
                chain.predict(qna=quessa)
                st.write(chain.predict(qna=quessa))
                print("entered here")

                #st.write(f"This is iteration {i}", i=i)



if st.button("check concepts"):
        from langchain.chains import GraphCypherQAChain
        from langchain.graphs import Neo4jGraph
        from langchain.prompts.prompt import PromptTemplate

        CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher translator who understands the question in english and convert to Cypher strictly based on the Neo4j Schema provided and following the instructions below:
    1. Generate Cypher query compatible ONLY for Neo4j Version 5
    2. Do not use EXISTS, SIZE keywords in the cypher. Use alias when using the WITH keyword
    3. Use only Nodes and relationships mentioned in the schema
    4. Always enclose the Cypher output inside 3 backticks
    5. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a concept name use `toLower(c.name) contains 'neo4j'`
    6. Candidate node is synonymous to Person
    7. Always use aliases to refer the node in the query
    8. Cypher is NOT SQL. So, do not mix and match the syntaxes
    Schema:
    {schema}
    Samples:
    Question: How many concepts involve data storage and analysis, and are related to Distributed Computing?
    Answer: MATCH (c1:concept)-[:relatedto]->(c2:concept{id:'concept2'}), (c3:concept)-[:partof]->(c4:concept{id:'concept4'}) RETURN COUNT(DISTINCT c1)
    Question: Identify the concepts related to Hadoop and its hierarchical relationships.
    Answer: MATCH (c1:concept)-[:relatedto|partof*]->(c2:concept{id:'concept3'}) RETURN DISTINCT c1.conceptname

    Question: 
    {question}

    Answer:
    """
        CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["schema", "question"], 
        template=CYPHER_GENERATION_TEMPLATE
        )

        graph = Neo4jGraph(
        url=url_neo, 
        username='neo4j', 
        password=url_pw
        )
        chain = GraphCypherQAChain.from_llm(
            llm, graph=graph, verbose=True,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            return_intermediate_steps=True
        )
    

        chain.invoke({"id":"1","schema":graph.schema,"question": "list the entities","query":"list the entities"})