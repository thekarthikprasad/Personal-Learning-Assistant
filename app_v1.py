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

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain.agents import load_tools
import os
from crewai import Agent, Task, Crew, Process
import os
from utils import read_pdf
from dotenv import load_dotenv,dotenv_values
from langchain_text_splitters import NLTKTextSplitter
from pathlib import Path
import nltk

load_dotenv()
llm_calls = 0
groq_key_m = os.environ['GROQ_API_KEY_M']
groq_key_l = os.environ['GROQ_API_KEY_L']
url_neo =os.environ['NEO4J_URL']
url_pw = os.environ['NEO4J_PASS']

if 'quess' not in st.session_state:
    st.session_state.quess = None

if 'a' not in st.session_state:
    st.session_state.a = None

if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

if 'concept_index' not in st.session_state:
    st.session_state.concept_index = 0

if 'answers' not in st.session_state:
    st.session_state.answers = []

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

if 'concepts' not in st.session_state:
    st.session_state.concepts = False

if 'ques' not in st.session_state:
    st.session_state.ques=[]

if 'concepts_extracted' not in st.session_state:
    st.session_state.concepts_extracted=[]

if 'button_click' not in st.session_state:
    st.session_state.button_click=False

if 'evaluations' not in st.session_state:
    st.session_state.evaluations=[]

if 'kg' not in st.session_state:
    st.session_state.kg = {}


llm = ChatGroq(temperature=0, groq_api_key=groq_key_m, model_name="mixtral-8x7b-32768")

st.title("Assist ✨")

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

def save_uploaded_file(uploaded_file, directory):

    """Manage uploads"""
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    posix_file_path = Path(file_path).as_posix()
    return posix_file_path


file_content = None
saved_file_path = None


uploaded_file = st.file_uploader("Choose a file", type=['pdf'])

if uploaded_file is not None:
    file_details = {
        "FileName": uploaded_file.name,
        "FileType": uploaded_file.type,
        "FileSize": uploaded_file.size
    }
    st.write(file_details)
    saved_file_path = save_uploaded_file(uploaded_file, "./data")
    st.write(f"File saved to {saved_file_path}")
    file_content = read_pdf.extract_text_from_pdf(saved_file_path)
    st.write(file_content)
else:
    st.write("Please upload a file to proceed.")


def get_prop_str(prop_dict, _id):
    """Generate Cypher"""
    s = []
    for key, val in prop_dict.items():
      if key != 'label' and key != 'id':
         s.append(_id+"."+key+' = "'+str(val).replace('\"', '"').replace('"', '\"')+'"')
    return ' ON CREATE SET ' + ','.join(s)

def get_cypher_compliant_var(_id):

    s = "_"+ re.sub(r'[\W_]', '', _id).lower() #avoid numbers appearing as firstchar; replace spaces
    return s[:20] #restrict variable size

def generate_cypher(file_name, in_json):
    """Generate Cypher using extracted json"""
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
    """Convert result to json"""
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

    connectionUrl = os.get_environ['NEO4J_URL']
    username ="neo4j"
    password = os.get_environ['NEO4J_PASS']

    driver = GraphDatabase.driver(connectionUrl, auth=(username, password))
    driver.verify_connectivity()

    

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


def callback():
    """Manage nested buttons"""
    st.session_state.button_click=True




def initialize_rag(file_path):
    """Initialize the retriever"""

    file_path = str(file_path)
    loader = PyPDFLoader(file_path)
    pages_1 = loader.load()
    vectorstore = Chroma.from_documents(
        documents=pages_1,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()
    return retriever

# Initialize RAG retriever only once

# Function to generate a question using RAG
def generate_question(concept):
    """Generate questions from the given document"""
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


def generate_actual(question):
    after_rag_template = """Answer only from the context
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
    actual = after_rag_chain.invoke(f"Answer these questions: {question}")
    return actual


def get_concepts(file_content):
    """ Generate the key concept entities in the text """

    knowledge_graphs = {
    'entities':[],
    'relationships':[]
    }
    memory = ConversationBufferMemory()
    text_splitter = NLTKTextSplitter(chunk_size=3000)
    texts = text_splitter.split_text(file_content)
    length = len(texts)
    wait = len(file_content)/3000
    st.write("your wait time is : ",wait)
    identified_concepts = [] 
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
        kg = merge_2_jsons(concep,knowledge_graphs)
        
        time.sleep(60)

    for i in knowledge_graphs['entities']:
            identified_concepts.append(i['conceptname'])
    print(identified_concepts)

    return identified_concepts, knowledge_graphs

import networkx as nx
import matplotlib.pyplot as plt

def create_knowledge_graph(kg):
    """Create knowledge graph of concepts using extracted entities and their relationships"""
    G = nx.DiGraph()

    for entity in kg['entities']:
        G.add_node(entity['id'], label=entity['conceptname'])
    
    # Add edges based on relationships
    for relationship in kg['relationships']:
        source, relation, target = relationship.split(' | ')
        G.add_edge(source, target, label=relation)
    
    return G

def visualize_knowledge_graph_matplotlib(G):
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'label')
    node_labels = nx.get_node_attributes(G, 'label')
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, labels=node_labels, node_color='skyblue', node_size=2000, edge_color='gray', font_size=15, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
    plt.title("Knowledge Graph")
    st.pyplot(plt.gcf())

from pyvis.network import Network

def visualize_knowledge_graph_pyvis(G):
    """pyvis graph visualization of the concept graph"""
    net = Network(height="750px", width="100%", directed=True)
    
    for node in G.nodes(data=True):
        net.add_node(node[0], label=node[1]['label'])
    
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], title=edge[2]['label'])
    
    return net

def find_weak(ent,scores):
    """Find scores <=5, to mark it weak
    ent: List
    scores: List
    """
    weaks=[]
    
    for i in scores:
        if i<=5:
            idx = scores.index(i)
            c = ent[idx]
            weaks.append(c)
    return weaks
        
def find_areas(G,conc):
        """Find weak areas of understanding using the concept graph G
        G: networkx graph
        Conc: concept (String)
        """
        
        if conc in G.nodes:
            pred = list(G.predecessors(conc))
            return pred
    
        
        


def evaluate_qs(q,a):
    """Evaluate the answers using agentic workflow
    q: question
    a: answer
    """

    llm = ChatGroq(temperature=0, groq_api_key=os.environ['GROQ_API_KEY_L1'], model_name="llama3-70b-8192")
    actual = generate_actual(q)
    print(actual)
    evaluator_1 = Agent(
    role='Check Answer',
    goal=f"""You are an expert teacher, you are expert in evaluating the answers and scoring them. You assign marks to the answers. A correct answers gets 1 mark, 
    You have question, user answer and the actual answer for that question. Use the actual answer for your reference for evaluating.
    a wrong answer gets a 0 and less approprite answer can have marks in the range 0 to 1. 
    The questions, answers and actual answer are in order: "questions:"{q},"\n" "user answers:"{a},"\n" "actual answers:"{actual}""",
    backstory="You are an expert in analysing the student answers. Thoroughly go through user answers, questions and actual answer and evaluate",
    verbose=True,
    allow_delegation=False,
    llm=llm
    )

    evaluator_2 = Agent(
    role='Check Answer',
    goal=f"""You are an expert teacher, you are expert in evaluating the answers and scoring them. You assign marks to the answers. A correct answers gets 1 mark, a wrong answer gets a 0 and less approprite answer can have marks in the range 0 to 1. You use
    your own knowledge to score the answers. Dont use actual answer. Use question and user answer and evaluate""",
    backstory="You are an expert in analysing the student answers",
    verbose=True,
    allow_delegation=False,
    llm=llm
    )

    eval_1 = Task(
    description="evaluate questions and assign marks for every question. Assign marks to every answer",
    agent=evaluator_1,
    expected_output="Finally result should be mark. compulasrily an integer: marks_awarded/total_marks."
    )

    eval_2 = Task(
    description="evaluate questions and assign marks for every question. Assign marks to every answer.",
    agent=evaluator_2,
    expected_output="Finally result should be mark. compulasrily an integer: marks_awarded/total_marks."
    )

    agg = Task(
    description="Check for the fairness of evaluation, and provide average of both evaluators, which is the final",
    agent=evaluator_2,
    expected_output="Now individual marks are changes, so does the total.. If any changes in total, update it. Finally result should be mark. Output should be average of evaluator_1 and evaluator_2. Output should only be a number no unecessary things in it. Eg: 11.5"
    )

    crew = Crew(
    agents=[evaluator_1,evaluator_2],
    tasks=[eval_1,eval_2,agg],
    verbose=2,
    process=Process.sequential,
    )

    result = crew.kickoff({"q":q, "a":a,"actual":actual})
    print(result)
    return float(result)


if st.button("Start Evaluation", on_click=callback):
    st.session_state.button_click = True

if not st.session_state.concepts:
    concepts,kg = get_concepts(file_content)
    st.session_state.concepts_extracted = concepts
    st.session_state.kg = kg
    st.session_state.concepts = True

if st.session_state.retriever is None:
    st.session_state.retriever = initialize_rag(saved_file_path)

if st.session_state.button_click:

    if st.session_state.concept_index <= len(st.session_state.concepts_extracted)+1:
        if not st.session_state.current_question:
            curr = st.session_state.concepts_extracted[st.session_state.concept_index]
            st.session_state.current_question = generate_question(curr)
            
        
        st.write(st.session_state.current_question)
        st.session_state.current_answer = st.text_input("Enter your answers", key="answer")

        if (st.button("Submit your answers", on_click = callback())):
            score = evaluate_qs(q=st.session_state.current_question, a=st.session_state.current_answer)
            print(score)
            
            st.session_state.evaluations.append(score)
            print(st.session_state.evaluations)
            #st.session_state.current_question = None
            st.session_state.concept_index+=1
            st.session_state.current_answer = ""
            st.write("breate in.... breathe out..")
            if st.session_state.concept_index < len(st.session_state.concepts_extracted):
                curr = st.session_state.concepts_extracted[st.session_state.concept_index]

                st.session_state.current_question = generate_question(curr)
                print("generated")
                st.write("next question in ")
                st.session_state.button_click = False
            else:
                st.session_state.current_question = None

    #st.experimental_rerun()
    
    st.session_state.button_click = False
    
    # ---------------- NEXT BUTTON (NOT WORKING -----------------------------------------#
                #st.write(st.session_state.current_question) 
                #st.write(st.session_state.current_question)
                #st.write(st.session_state.current_question)
            

            #st.session_state.button_click = False
    
        # st.write(st.session_state.current_question)
        # st.session_state.current_answer = st.text_input("Enter your answers", key="answer1")
        # if st.session_state.current_answer:
        #     evaluate_qs(q=st.session_state.current_question,a=st.session_state.current_answer)
        # st.session_state.current_answer=None

        # if (st.button("Next ->",on_click=callback())):
        #     #evaluate_qs(q=st.session_state.current_question, a=st.session_state.current_answer)    
        #     st.session_state.concept_index+=1
        #     st.session_state.current_answer = ""
            
        #     if st.session_state.concept_index < len(st.session_state.concepts_extracted):
        #         curr = st.session_state.concepts_extracted[st.session_state.concept_index]
        #         st.session_state.current_question = generate_question(curr)
        #         #st.write(st.session_state.current_question)
        #         st.session_state.current_answer = "" 
        #         st.write(st.session_state.current_question)
        #         #st.write(st.session_state.current_question)
        #     else:
        #         st.session_state.current_question = None

    #----------------------------------------------------------------------------------------#
                

if st.session_state.concept_index >= len(st.session_state.concepts_extracted):
                st.write("You have completed all questions.")
                st.write("Your answers:", st.session_state.answers)
st.session_state.button_click = True




if st.button("Results"):
    import tempfile

    kg = st.session_state.kg

    G = create_knowledge_graph(kg)
    print(G.nodes)

    net = visualize_knowledge_graph_pyvis(G)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
        temp_html_path = f.name
        net.save_graph(temp_html_path)

    with open(temp_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
        
    st.components.v1.html(html_content, height=800, width=800)
    
    # Remove the temporary file
    os.remove(temp_html_path)

    st.write(find_weak(st.session_state.concepts_extracted,st.session_state.evaluations))
    c = find_weak(st.session_state.concepts_extracted,st.session_state.evaluations)[8]
    st.write(find_areas(G,c))





