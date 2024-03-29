import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec


sentences = [
    'I love my dog', 
    'I love my cat'
    ]

sentences = ['In February 1969, Hollywood actor Rick Dalton, star of 1950s TV Western series Bounty Law, fears his career is fading, with his recent roles being guest appearances as villains. His agent Marvin Schwarz advises him to make Spaghetti Westerns in Italy, which Dalton considers beneath him. Daltons best friend and stunt double, Cliff Booth – a war veteran who lives in a trailer with his pit bull, Brandy – drives Dalton around due to his DUI arrests and drivers license suspension. Booth struggles to find stunt work because of rumors he murdered his wife. Actress Sharon Tate and her husband, director Roman Polanski, have moved next door to Dalton, and Dalton dreams of befriending them to revive his career. That night, Tate and Polanski attend a celebrity-filled party at the Playboy Mansion. The next day, Booth recalls a sparring contest he had with Bruce Lee on the set of The Green Hornet resulting in Booths firing. Meanwhile, Charles Manson stops by the Polanski residence looking for Terry Melcher, who used to live there, but is turned away by Jay Sebring. As Tate runs errands, she stops at the Fox Bruin Theater to watch herself in The Wrecking Crew. Dalton is cast as the villain in the pilot for the TV Western Lancer and strikes up a conversation with eight-year-old co-star Trudi Frazer. During filming, Dalton struggles to remember his lines and suffers a breakdown in his trailer. He subsequently delivers a strong performance that impresses Frazer and the director, Sam Wanamaker. Booth picks up a hitchhiker, "Pussycat", and takes her to Spahn Ranch, where he once worked on the set of Bounty Law. He observes the many hippies living there. Suspecting they may be taking advantage of the ranchs elderly owner, George Spahn, Booth insists on checking on him despite "Squeaky"s objections. Booth speaks with the nearly blind Spahn, who dismisses his concerns. Upon leaving, Booth discovers that "Clem" has punctured a tire on Daltons car. Booth beats Clem and makes him change the tire. "Tex" is summoned to deal with the situation, but arrives as Booth is driving away. After watching Daltons guest performance on an episode of The F.B.I., Schwarz books him as the lead in Sergio Corbuccis Spaghetti Western Nebraska Jim. Dalton takes Booth with him for a six-month stint in Italy, during which he films three additional movies and marries Italian starlet Francesca Capucci. By the end of their stay, Dalton can no longer afford Booths services. Returning to Los Angeles on August 8, 1969, Dalton and Booth go out for drinks to commemorate their time together, then go back to Daltons house. Meanwhile, Tate and Sebring go out for dinner with friends, then return to Tates house. Booth smokes an LSD-laced cigarette purchased earlier and takes Brandy for a walk while Dalton prepares drinks. Tex, Sadie, Katie, and Flowerchild arrive outside in preparation to murder everyone in Tates house, but Dalton hears the cars muffler and orders them off the street. Recognizing him, the four change their plans and decide to kill him instead, after Sadie reasons that Hollywood has taught them to murder. Flowerchild deserts them, speeding off with their car. Breaking into Daltons house, they confront Booth and Capucci. Tex aims his pistol at Booth. Now high on the LSD, Booth chats with the intruders, remembering them from Spahn Ranch. Booth signals Brandy to attack Tex. Sadie lunges at Booth with a knife. Booth throws a can at her face and signals Brandy to attack her. Capucci punches Katie and runs away. Katie dives at Booth after he fights and kills Tex. Realizing a knife is stuck in his thigh, Booth kills Katie by smashing her face against household decor then passes out. A wounded and crazed Sadie stumbles outside into the pool firing Texs pistol. Dalton, floating in the pool, listening to music on headphones, oblivious to the chaos inside, is alarmed. He retrieves a flamethrower retained from his earlier film The 14 Fists of McClusky and incinerates Sadie. The police arrive and Booth later regains consciousness and is taken away in an ambulance. After promising to visit Booth in the hospital, Dalton is invited by Sebring and Tate to their house for a drink, which he accepts.', 'Killing and cars']

# Tokenize the sentences. The Word2Vec model requires that each sentence be a list of words.
# Here, we're splitting each sentence into words based on space.
tokenized_sentences = [sentence.lower().split() for sentence in sentences]

# Training a Word2Vec model with the tokenized sentences
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Finding the most similar words
try:
    similar_words = model.wv.most_similar('hollywood', topn=3)  # perhaps it might find 'dog'
    print(similar_words)
except KeyError as e:
    print(f"Error: {e}")

# tokenizer = Tokenizer(num_words = 100)
# tokenizer.fit_on_texts(sentences)
# word_index = tokenizer.word_index
# print(word_index)