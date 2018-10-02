#encoding: utf-8


import pandas as pd
import numpy as np
import glob
import gzip
import csv 
import tarfile
from contextlib import closing
from xml.etree import ElementTree as ET
import xml.etree.ElementTree as ET2
import xml.dom.minidom
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def read_file(filename):
  df = pd.read_csv(filename)
  df = df.drop(df[df.feed_source == "gnip_twitter"].index)
  df = df.dropna(how='all') 
  #df = df.drop(df[df.momentum == None ].index)
  return df.to_csv(filename, sep='\t')

def extractMoreover(filename):
  df = pd.read_csv(filename)
  df_id = pd.read_csv(filename, usecols = [5])
  

  for filepath in glob.glob("/Users/hualtinisik/Desktop/moreover/2018/03/01/*.gz"):
    with gzip.open(filepath,'r') as f:
      print filepath
      root = ET.parse(f).getroot()
      articles = root[4]
      #print articles
      for i in articles:
        article_id = i[1].text
        #problems!!!
        for cell in df_id.values:
          if str(int(cell[0])) == article_id:
            for index in range(len(df['id'].values)):
              if str(int(df['id'].values[index])) == article_id:
                text_content = i[4].text
                #print type(text_content.encode('utf-8'))
                df.ix[index, 'ARTICLE CONTENT'] = text_content.encode('utf-8')
                #df.set_value(index, 'text', text_content.encode('utf-8'))
                print "FOUND ID IS:", article_id, "AT INDEX", index, "OF CSV"
  df.to_csv('out.csv')
'''
      with gzip.open("/Users/hualtinisik/Desktop/moreover/2018/03/01/moreover-1519866001805.xml.gz",'r') as f1:
        root = ET.parse(f1).getroot()
        articles = root[4]
        for i in articles:
          article_id = i[1].text  
        if article_id not in df_id.values:
          print True
    
    '''
def extractOpoint(filename):
  import json
  df = pd.read_csv(filename)
  df_id = pd.read_csv(filename, usecols = [4])
  
  length = len("/Users/hualtinisik/Desktop/opoint/2018/03/01/")
  for filepath in glob.glob("/Users/hualtinisik/Desktop/opoint/2018/03/01/*.gz"):
    if "opoint-1519866574041.json.xml" in filepath[length:]:
      print "YOU HAVE READ 20 FILES SO FAR"
    if "opoint-1519888681989.json.xml.gz" in filepath[length:]:
      print "YOU HAVE READ 1/3 OF THE FILES"
    if "opoint-1519907463336.json.xml.gz" in filepath[length:]:
      print "YOU HAVE READ 1/4 OF THE FILES"
    with gzip.open(filepath,'r') as f:
      try:
        text = json.load(f)
        for document in text["searchresult"]["document"]:
          id_site = str(document["id_site"])
          id_article = str(document["id_article"])
          for index in range(len(df_id.values)):
            cell = df_id.values[index]
            indexUnderscore = str(cell[0]).index("_")
            beforeUnderscore = str(cell[0])[:indexUnderscore]
            afterUnderscore = str(cell[0])[indexUnderscore+1:]
            if beforeUnderscore == id_site and afterUnderscore == id_article:
              text_content = document["body"]["text"]
              print "FOUND A MATCH:", id_site, id_article, index, filepath
              #print type(text_content.encode('utf-8'))    THIS IS EQUAL TO str
              df.ix[index, 'ARTICLE CONTENT'] = text_content.encode('utf-8')
              #df.set_value(index, 'text', text_content.encode('utf-8'))
              
      except:
        print "ERROR AT FILE:", filepath
  df.to_csv('out.csv')

      

def extractOpointOLD():

  files = glob.glob("/Users/hualtinisik/Desktop/temp/opoint/2018/03/01/*.gz")
  length = len("/Users/hualtinisik/Desktop/temp/opoint/2018/03/01/")  
  for i in range(len(files)):
    index1 = files[i][length:].index('1')
    indexDot = files[i][length:].index('.')
    name = files[i][length:][index1:indexDot]
    f = gzip.open(files[i], 'r')
    file_content = f.read()
    
    with open("/Users/hualtinisik/Desktop/outputsOpoint/"+str(name)+".txt", "w+") as the_file:
      #myfile = the_file.readlines()
      the_file.write(file_content)
    f.close()
    
def readDocsOpoint():
    #print the_file.read().split("\n")[1:5]
    #['\t"searchresult": {\n', '\t\t"documents": 500,\n', '\t\t"first_timestamp": 1519862409,\n', '\t\t"last_timestamp": 1519621200,\n']
    #['\t"searchresult": {', '\t\t"documents": 500,', '\t\t"first_timestamp": 1519862409,', '\t\t"last_timestamp": 1519621200,']
    '''
    print array
    for j in range(len(array)):
      if j>0 and j<len(array)-1:
        if array[j-1] != array[j+1]:
          print len(array),j, "error"
    '''
    files = glob.glob("/Users/hualtinisik/Desktop/outputsOpoint/*.txt")
    length = len("/Users/hualtinisik/Desktop/outputsOpoint/")  
    for i in range(len(files)):
      name = files[i][length:]
      f = open(files[i], 'r')
      file_content = f.readlines()
      stack = 0
      #array = []
      #array.append(stack)
      count = 0
            
      with open("/Users/hualtinisik/Desktop/outputsOpointDocs/"+str(name), "w+") as the_file:
        for line in file_content:
          if "\"document\":" in line:
            count += 1
          if count == 1:
            the_file.write(line)
            if '[' in line:
              stack += 1
              #array.append(stack)
            if ']' in line:
              stack -= 1
              #array.append(stack)
            if stack == 0:
              count == 0      
        #myfile = the_file.readlines()
      f.close()
def readDocsMoreover():
  files = glob.glob("/Users/hualtinisik/Desktop/outputsMoreover/*.txt")
  length = len("/Users/hualtinisik/Desktop/outputsMoreover/")  
  for i in range(len(files)):
    name = files[i][length:]
    f = open(files[i], 'r')
    file_content = f.readlines()
    stack = 0
    #array = []
    #array.append(stack)
    count = 0
          
    with open("/Users/hualtinisik/Desktop/outputsMoreoverDocs/"+str(name), "w+") as the_file:
      for line in file_content:
        if "\"document\":" in line:
          count += 1
        if count == 1:
          the_file.write(line)
          if '[' in line:
            stack += 1
            #array.append(stack)
          if ']' in line:
            stack -= 1
            #array.append(stack)
          if stack == 0:
            count == 0      
      #myfile = the_file.readlines()
    f.close()
    
    
def checkOpoint():

  files = glob.glob("/Users/hualtinisik/Desktop/OpointOnlyDocs/*.txt")
  length = len("/Users/hualtinisik/Desktop/OpointOnlyDocs/")  
  found = 0
  for i in range(len(files)):
    f = open(files[i], 'r')
    file_content = f.readlines()
 
    for line in file_content:
      if "35955" in line:
        found += 1
  print found
  
    
def checkMoreover():
  more = pd.read_csv("moreover.csv")
  files = glob.glob("/Users/hualtinisik/Desktop/Moreover text files/*.txt")
  length = len("/Users/hualtinisik/Desktop/Moreover text files/")  
  found = 0
  for i in range(len(files)):
    f = open(files[i], 'r')
    file_content = f.readlines()
 
    for line in file_content:
      if "33434533037" in line:
        found += 1
        
  print found  
    
def main():
  #print read_file("data_drop_w_momentum.csv")
  #extractMoreover()
  #extractOpointOLD()
  #readDocsOpoint()
  #readDocsMoreover()
  #checkMoreover()
  #extractMoreover("moreover.csv")
  extractOpoint("opoint.csv")
  
  
if __name__ == "__main__":
    main()
