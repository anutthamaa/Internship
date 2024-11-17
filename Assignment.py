#import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
from nltk.tokenize import word_tokenize, sent_tokenize
import asyncio
import aiohttp
from nltk.corpus import cmudict
pronouncing_dict = cmudict.dict()
import re
#nltk.download('all')

## To create new columns in the dataframe
def createNewColumns(df):
    df['POSITIVE SCORE'] = 0  
    df['NEGATIVE SCORE'] = 0  
    df['POLARITY SCORE'] = 0  
    df['SUBJECTIVITY SCORE'] = 0 
    df['AVG SENTENCE LENGTH'] = 0 
    df['PERCENTAGE OF COMPLEX WORDS'] = 0 
    df['FOG INDEX'] = 0
    df['AVG NUMBER OF WORDS PER SENTENCE'] = 0
    df['COMPLEX WORD COUNT'] = 0
    df['WORD COUNT'] = 0
    df['SYLLABLE PER WORD'] = 0
    df['PERSONAL PRONOUNS'] = 0
    df['AVG WORD LENGTH'] = 0

## To create a list of stop words
def stopwordsDict():
    stopwords_path = 'stopwords'
    stopwords_file = os.listdir(stopwords_path)
    stop_words = []

    for file in stopwords_file:
        if file.endswith('.txt'):
            file_path = os.path.join(stopwords_path, file)
    
            with open(file_path, 'r') as f:
                stop_words.extend(f.read().strip().lower().splitlines())
    return stop_words


## To create a list of positive and negative words
def createMasterDictWords(filename, stop_words):
    masterdict_path = 'master_dictionary'
    masterdict_files = os.listdir(masterdict_path)
    words = []
    
    for file in masterdict_files:
        if file == filename:
            file_path = os.path.join(masterdict_path, file)
    
            with open(file_path, 'r') as f:
                master_dictionary = f.read().strip().lower().splitlines()
        
                for word in master_dictionary:
                    if word.lower() not in stop_words:
                        words.append(word.strip())
    return words

## To calculate positive and negative store
def calculatePosNegScore(tokens, words, symbol):
    score = 0
    for token in tokens:
        if (symbol == '+'):
            if (token.lower() in words):
                score += 1
        elif (symbol == '-'):
            if (token.lower() in words):
                score -= 1
    return abs(score)

## To calculate polarity score
def calculatePolarityScore(pos_score,neg_score):
    polarity_score = round((pos_score - neg_score)/ ((pos_score + neg_score) + 0.000001),2)
    return polarity_score

## To calculate subjectivity score
def calculateSubjectivityScore(pos_score,neg_score,word_count):
    subjectivity_score = round((pos_score + neg_score)/ ((word_count) + 0.000001),2)
    return subjectivity_score

## To check whether the word is having more than 2 syallables
def count_syllables(word):
    if word.lower() in pronouncing_dict:
        return max([len(list(y for y in x if y[-1].isdigit())) for x in pronouncing_dict[word.lower()]])
    else:
        return 1

## To count the vowels in a word
def count_vowels(word):
    exceptions = ["es", "ed"]      
    vowels = "aeiouAEIOU"
    vowel_count = 0
    
    for char in word:
        if char in vowels:
            vowel_count += 1
        
    for exception in exceptions:
        if word.lower().endswith(exception):
            vowel_count = vowel_count - 1
    return vowel_count

## To remove the punctuation marks except the alphabets and numbers
def remove_other_char(content):
    content = re.sub('[^a-zA-Z0-9]',' ',content)
    return content

## To remove stop words in a text
def remove_stopwords(content,stop_words):
    clean_data = []
    for i in content.split():
        if i.strip().lower() not in stop_words:
            clean_data.append(i.strip().lower())
    return " ".join(clean_data)

## To count the pronouns present in a text
def personalPronouns(content):
    pattern = r'\b(?:I|we|my|ours|us)\b'
    matches = re.findall(pattern, content, flags=re.IGNORECASE)
    personal_pronouns = [match for match in matches if match != 'US']
    pronoun_count = len(personal_pronouns)
    return pronoun_count

## Preprocess the raw text to remove stopwords and punctuation marks
def data_cleaning(content, stop_words):
    content = remove_other_char(content)
    content = remove_stopwords(content, stop_words)
    return content

## Scrapes the title and content from the URL provided
async def scrape_url(session, url, url_id, output_directory):
    try:
        title = ''
        article_content = ''
        async with session.get(url) as response:
            if response.status == 404:
                title = 'Title not found'
                article_content = 'Page not found'
            else:
                webpage = await response.text()
                soup = BeautifulSoup(webpage,'lxml')
                if (soup.find_all('h1')):
                    title = soup.find_all('h1')[0].text
                else:
                    title = 'Title not found'
            
                for j in soup.find_all('div', class_='td-post-content'):
                    for pre in j.find_all('pre'):
                        pre.extract() 
                        article_content = j.text
                
        output_file = os.path.join(output_directory, f'{url_id}.txt')
        with open(output_file, 'w', encoding='utf-8') as text_file:
            text_file.write(f'{title}\n\n')
            text_file.write(f'{article_content}')
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")

async def main(df, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for index, row in df.iterrows():
            url = row['URL']
            url_id = row['URL_ID']
            task = scrape_url(session, url, url_id, output_directory)
            tasks.append(task)

        await asyncio.gather(*tasks)

## Calculates all the scores and other metrics and stores it back to dataframe
def textPreProcessing(article_files, df, stop_words):
    for filename in article_files:
        if filename.endswith('.txt'):
            file_path = os.path.join(output_directory, filename)
            index = df[df.URL_ID == float(filename.rstrip('.txt'))].index[0]
        
            with open(file_path, 'r', encoding='utf-8') as text_file:
                article_text = text_file.read()
            
                sentences = sent_tokenize(article_text)
                filtered_text = data_cleaning(article_text, stop_words)
                tokens = word_tokenize(filtered_text)             
                
                ## EXTRACTING DERIVED VARIABLES
                positive_score = df.at[index,'POSITIVE SCORE'] = calculatePosNegScore(tokens, positive_words, '+')
                negative_score = df.at[index,'NEGATIVE SCORE'] = calculatePosNegScore(tokens, negative_words, '-')
                df.at[index,'POLARITY SCORE'] = calculatePolarityScore(positive_score,negative_score)
                df.at[index,'SUBJECTIVITY SCORE'] = calculateSubjectivityScore(positive_score,negative_score,len(tokens))

                ## COMPLEX WORD COUNT
                complex_word_count = df.at[index,'COMPLEX WORD COUNT'] = sum(1 for word in tokens if count_syllables(word) > 2)

                ## ANALYSIS OF READABILITY
                df.at[index, 'AVG SENTENCE LENGTH'] = len(sentences)
                complex_words_percent = df.at[index,'PERCENTAGE OF COMPLEX WORDS'] = round(complex_word_count/len(tokens),2) 
                df.at[index,'FOG INDEX'] = round(0.4 * (len(sentences) + complex_words_percent),2)

                ## AVERAGE NUMBER OF WORDS PER SENTENCE
                df.at[index,'AVG NUMBER OF WORDS PER SENTENCE'] = round(len(tokens)/len(sentences))

                ## WORD COUNT
                df.at[index,'WORD COUNT'] = len(tokens)

                ## SYLLABLE COUNT PER WORD
                df.at[index,'SYLLABLE PER WORD'] = sum([count_vowels(word) for word in tokens])

                ## PERSONAL PRONOUNS
                df.at[index, 'PERSONAL PRONOUNS'] = personalPronouns(article_text)

                ## AVERGAGE WORD LENGTH         
                total_characters = sum(len(word) for word in tokens)
                df.at[index,'AVG WORD LENGTH'] = round(total_characters/len(tokens),2)
            

if __name__ == '__main__':
    inputFile_path = 'Input.xlsx'
    blackCoffer_df = pd.read_excel(os.path.join(inputFile_path))
    createNewColumns(blackCoffer_df)

    stopWords = stopwordsDict()
    positive_words = []
    positive_words = createMasterDictWords('positive-words.txt', stopWords)
    negative_words = []
    negative_words = createMasterDictWords('negative-words.txt', stopWords)

    output_directory = 'extracted_articles'
    asyncio.run(main(blackCoffer_df, output_directory))

    article_files = os.listdir(output_directory)
    textPreProcessing(article_files,blackCoffer_df, stopWords)
    
    excel_file = 'Output.xlsx'
    blackCoffer_df.to_excel(excel_file, index=False)

    print('Text preprocessing done and exported to Output.xlsx file!!!')




    