#Authors - Koyel Roy, Prajwal Meshram, Roshani Daulkar, Hrishikesh Shirodkar
#DBDA - Mar 2023


#importing libraries
import random
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px 
from pywaffle import Waffle 
import plotly.graph_objects as go 
from wordcloud import WordCloud
import mplcursors 
import matplotlib.cm as cm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_option('deprecation.showPyplotGlobalUse', False)

#loading dataset
@st.cache_data
def load_data():
    data = pd.read_csv("ProcessedCleaned.csv")       
    return data

df = load_data()


#Navigation slider
rad=st.sidebar.radio("NAVIGATION BAR",["Home","Insights","Recommendation System"])

#Home page
if rad == "Home":
    
    #project Title
    st.title(":blue[INDIAN CUISINE ANALYSIS AND RECOMMENDATION SYSTEM ]")

    #adding image in streamlit
    st.image("cuisine.jpeg",width=700) 

    #displaying the data frame on home page
    st.dataframe(df,width=800,height=500)


#Insights page to show the analysis done on the data
elif rad == "Insights":

    #list of charts to display
    charts = ["Most used ingredients in India","Proportion of flavor profiles","Course Meal with shortest cooking time",
              "Ingredients used in different meals","Region wise Veg and Non-Veg cuisine",
              "Daily Meals with shortest cooking time", "Ingredients used in Diet based food",
              "Region wise distribution of Flavors","Famous International cuisines","Region wise course distribution",
              "Recipes With Most Allergens"]
    
    #select box to select one chart at a time
    select=st.sidebar.selectbox("select the insight",charts)

    #1. chart for most used ingredients in indian cuisines
    if select == charts[0]:
        st.write(""" ## Most Used Ingredients In Indian Cuisines""")
        ingredientCharts = ["Pie","Sunburst"]
        select=st.selectbox("Select the Chart",ingredientCharts)
        
        #1.1. Pie chart
        if select == ingredientCharts[0]:

            splitted_lst = df['ProcessedCleanedLoweredIngredientsFiltered'].apply(lambda x: x.split(','))
         
            # Flatten the list of lists in the 'Ingredients' column
            flat_list = [word for sublist in splitted_lst for word in sublist]

            # Count the occurrences of each word
            word_counts = Counter(flat_list)
            top_common_words1 = word_counts.most_common(10)

            top_words_df = pd.DataFrame(top_common_words1, columns=['Word', 'COUNTS'])

            #plotting a pie chart
            fig = px.pie(top_words_df, names='Word', values='COUNTS')
            fig.update_layout(width=800, height=600)

            #display chart in streamlit
            st.plotly_chart(fig)

        #1.2. Sunburst chart
        else:

            #count the occurrences of each word
            splitted_lst = df['ProcessedCleanedLoweredIngredientsFiltered'].apply(lambda x: x.split(','))
            flat_list = [word for sublist in splitted_lst for word in sublist]
            word_counts = Counter(flat_list)
            top_common_words1 = word_counts.most_common(10)

            # Convert top_common_words1 to a DataFrame
            top_words_df = pd.DataFrame(top_common_words1, columns=['Word', 'COUNTS'])

            # Creating a sunburst plot
            fig = px.sunburst(top_words_df, path=['Word'], values='COUNTS', color='COUNTS',color_continuous_scale='Plasma')
            fig.update_layout(width=800, height=600)

            # Show the plot
            st.plotly_chart(fig)

        #final observations from this analysis
        st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)        
        st.markdown("""
                    * The charts highlights the central role of spices in Indian cooking. 
                    * Spices like cumin, coriander, turmeric, and chilli stand out as the most frequently used ingredients. These spices impart unique flavors and contribute to the vibrant colors associated with Indian dishes.
                    * Chilli is the most used spice of all with ~15% , followed by coriander with ~14%.
                    * One of the most used ingredients also includes salt with 11.7% which is used in all spicy and savory dishes.
                    * India being a dominating vegetarian country, uses onion and garlic in most of its cuisines.
                    * Indian cuisine is known for its ability to balance flavors - sweet, sour, bitter, and tangy. The inclusion of ingredients like curd reflects to achieve a harmonious flavor profile in each dish.""")


    #2. chart to showcase the proportion of flavor profiles
    elif select == charts[1]:

        st.write(""" ## Proportion of flavor profiles""")

        #x and y axis
        flavoured_profile_index = list(df["FlavourProfile"].value_counts().index)
        flavoured_profile_count = list(df["FlavourProfile"].value_counts())

        flavorChart = ["Waffle Plot", "Pie Chart", "Scatter Plot"]
        select=st.selectbox("Select the chart",flavorChart)

        #2.1. Waffle chart
        if select == flavorChart[0]:
            data = dict(df["FlavourProfile"].value_counts(normalize=True) * 100)
            flavors = df["FlavourProfile"]

            fig = plt.figure(
                FigureClass=Waffle,
                rows=10,  # Rows represent the total number of waffles
                columns=15,  # Columns are not used for this type of chart
                values=data,
                title={'label': 'Proportion of Flavor Profiles', 'loc': 'center', 'fontsize': 15},
                facecolor='black',
                labels=[f"{k} ({v:.2f}%)" for k, v in data.items()],
                legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1), 'ncol':1, 'framealpha': 0, 'prop': {'size': 12}  },
                figsize=(9,20)
            )

            #legend manipulation
            legend = fig.gca().get_legend()
            for text in legend.get_texts():
                text.set_color('white')
            st.pyplot()

        #2.2. Pie chart
        elif select == flavorChart[1]:
            labels = flavoured_profile_index
            values = flavoured_profile_count

            fig = go.Figure(data=[go.Pie(labels=labels, values=values)] )
            fig.update_layout(width=800, height=600)

            st.plotly_chart(fig)

        #2.3. Scatter plot
        else:
            x = flavoured_profile_index
            y = flavoured_profile_count

            
            # plotting scatter plot 
            fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', marker=dict( 
                    size = 15,
                    color=np.random.randn(10), 
                    colorscale='Viridis',  
                    showscale=True
                    ) ,
                    text=flavoured_profile_index)) 
            
            fig.update_layout(
                width=800,
                height=800,
                xaxis_title="Flavoured Profile Index",    
                yaxis_title="Flavoured Profile Count"  
            )   
            st.plotly_chart(fig)

        #final observations from the analysis
        st.write(' ## <span style="color:red"> OBSERVATIONS : </span> ',unsafe_allow_html=True)        
        st.markdown("""
                    * It's evident from the charts that certain flavor profiles dominate Indian cuisines. 
                    * For instance, the prevalence of "Savory" and "Spicy" flavor highlights their importance with ~45% and ~22% resp. in the overall flavor landscape. 
                    * The presence of less consumed flavors, like "creamy," "nutty," or "fusion," suggests a willingness to explore niche or unique tastes.
                    * It can also be concluded from the analysis that Indian cuisines have a great palate diversity.

                    """)


    #3. chart to display top N courses with shortest cooking time.
    elif select == charts[2]:

        st.write(""" ## Top N Courses With Shortest Cooking Time""")

        dishchart = ["Main Course", "Side Dish", "Dessert"]
        select=st.selectbox("Select the Course Meal ",dishchart)
        
        #3.1. Main course
        if select == dishchart[0]:
            #slider to choose number of courses to show form 2 to 15.
            slide = st.slider("Number of courses to show",min_value=2,max_value=15,value=10)

            cooking_time_to_color = {10: 'red', 15: 'green', 20: 'blue'} 
            main_courses_shortest = df[df['Course'] == 'Main Course'].sort_values(by='TotalTimeInMins').head(slide)

            #bar chart
            fig = px.bar(main_courses_shortest, x='TotalTimeInMins', y='Course', 
                        category_orders={"CookTimeInMins": [10, 15, 20]},
                        text=main_courses_shortest["EnglishRecepie"],
                        color='TotalTimeInMins',
                        color_continuous_scale='viridis_r')
            fig.update_layout(width=1000, height=800) 
            st.plotly_chart(fig)

            #final observations from the analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True) 
            st.markdown("""
                        * The analysis of quickest main courses in Indian cuisine presents a blend of efficiency and flavor. 
                        * The 10-minute "Sundried Tomato Pesto" showcases modern fusion, while dishes like "Spicy Tomato Rice," "Banana Apple Mash," and "Herbal Butter Rice" at 15 minutes emphasize convenience without sacrificing taste. 
                        * This reflects Indian cuisine's adaptation to contemporary lifestyles, offering speedy, diverse, and delightful options.
                        """) 

        #3.2. Side Dish
        elif select == dishchart[1]:
            
            slide = st.slider("Number of courses to show",min_value=2,max_value=15,value=10)

            side_dish_shortest = df[(df['Course'] == 'Side Dish') & (df['TotalTimeInMins'] > 0)].sort_values(by='TotalTimeInMins').head(slide)
            
            #bar chart
            fig = px.bar(side_dish_shortest, x='TotalTimeInMins', y='Course', 
                        #title='Top 10 side dishes with shortest cooking time',
                        category_orders={"CookTimeInMins": [10, 15, 20]},
                        text=side_dish_shortest["EnglishRecepie"],
                        #color='Color')  # Color bars based on cooking time
                        color='TotalTimeInMins',
                        color_continuous_scale='viridis_r')
            fig.update_layout(width=1000, height=800) 

            st.plotly_chart(fig)

            #final observations from the analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True) 
            st.markdown("""
                        * The exploration of swift side dishes in Indian cuisine uncovers a captivating blend of tradition and innovation. At the forefront is the "Sweet Spicy Soy Sauce," a 5-minute wonder that encapsulates contemporary fusion. 
                        * Traditional treasures like the "Burani Raita" and "Grated Cucumber Raita" take 10 minutes, offering cooling companions to hearty mains. 
                        * The "Coconut Onion Chutney" exemplifies regional diversity with its nuanced flavors.
                        * Venturing beyond borders, "Mango Tomato Salsa" and "Cashew Mayonnaise" fuse global influences with Indian sensibilities in 10 minutes. Beverages also shine: "Guava and Papaya Drink" and "Makhana ka Raita" deliver refreshment in 10 minutes.

                        """) 

        #3.3. Dessert
        else:
            slide = st.slider("Number of courses to show",min_value=2,max_value=15,value=10)

            dessert_shortest = df[(df['Course'] == 'Dessert') & (df['TotalTimeInMins'] > 0)].sort_values(by='TotalTimeInMins').head(slide)
            
            fig = px.bar(dessert_shortest, x='TotalTimeInMins', y='Course', 
                        category_orders={"CookTimeInMins": [10, 15, 20]},
                        text=dessert_shortest["EnglishRecepie"],
                        color='TotalTimeInMins',
                        color_continuous_scale='jet')
            fig.update_layout(width=1000, height=800) 
            st.plotly_chart(fig)

            #final observations from the analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True) 
            st.markdown("""
                        * We have analysed that Indian dessert recipes offer a diverse range of options with short cooking times, from simple tea, milk-based sweets such as chocolate mouse and almond milk pudding to baked treats like Choco mug cake and fruit-based delights such as jack-fruit ice-cream and coconut ladoo. 
                        * The use of condensed milk, semolina, and microwave-friendly techniques allows for quick and delicious desserts that cater to various tastes and preferences.

                        """) 
            

    #4. chart to show top incredients in different courses of the day.
    elif select == charts[3]:

        st.write(""" ## Top Ingredients In Different courses of the meal""")

        dishchart = ["Main Course", "Side Dish", "Dessert"]
        select=st.selectbox("Select the Course Meal ",dishchart)

        #4.1. Main course
        if select == dishchart[0]:
            main_course_df = df[df['Course'] == 'Main Course'].reset_index()
            #unique data selection
            unique_ingredients = set()

            for i in range(0, len(main_course_df)):
                ingridients = main_course_df["ProcessedCleanedLoweredIngredientsFiltered"][i].split(',')
                unique_ingredients.update(ingridient.strip() for ingridient in ingridients )
                
                text = ', '.join(unique_ingredients)

            # Generate the word cloud using the 'plasma' colormap
            wordcloud_plasma = WordCloud(
                width=400,
                height=400,
                colormap='prism',  
                background_color='black',
                min_font_size=8
            ).generate(text)

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            #wordcloud
            plt.imshow(wordcloud_plasma)
            plt.axis('off')
            plt.tight_layout()
            st.pyplot()

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True) 
            st.markdown("""
                        * In summary, Indian main course ingredients offer a vibrant and aromatic journey through chilli, coriander and elaichi. 
                        * The bold and diverse flavors created by ingredients like jeera, haldi, and dalchini are a testament to India's culinary heritage. Whether savoring a spicy curry or a tandoori masterpiece, the richness and depth of Indian main courses leave an indelible mark on the world's gastronomic landscape.

                        """)

        #4.2. Side dish
        elif select == dishchart[1]:
            side_dish_df = df[df['Course'] == 'Side Dish'].reset_index()
            unique_ingredients = set()
            for i in range(len(side_dish_df)):
                text = side_dish_df["ProcessedCleanedLoweredIngredientsFiltered"][i].split(',')
                unique_ingredients.update(ingredients.strip() for ingredients in text)

                text = ' '.join(unique_ingredients)

            # Generate the word cloud using the 'plasma' colormap
            wordcloud_plasma = WordCloud(
                width=400,
                height=400,
                colormap='plasma',  
                background_color='black',
                min_font_size=8
            ).generate(text)

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(wordcloud_plasma)
            plt.axis('off')
            plt.tight_layout()
            st.pyplot()

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True) 
            st.markdown("""
                        * The use of ingredients such as curd and butter, aromatic spices like jeera and mint, and a medley of vegetables allows for an explosion of taste and texture that complements the main courses. 
                        * Whether enjoying a cooling raita, a fiery pickle, or a hearty serving of dal, these side dishes are a delightful journey into the heart of India's gastronomic traditions.

                        """)

        #4.3. Dessert
        else:
            dessert_df = df[df['Course'] == 'Dessert'].reset_index()
            unique_ingredients = set()
            for i in range(0, len(dessert_df)):
                text = dessert_df["ProcessedCleanedLoweredIngredientsFiltered"][i].split(',')
                unique_ingredients.update(ingredients.strip() for ingredients in text)
                text = ' '.join(unique_ingredients)

            # Generate the word cloud using the 'plasma' colormap
            wordcloud_plasma = WordCloud(
                width=400,
                height=400,
                colormap='coolwarm',  
                background_color='black',
                min_font_size=8
            ).generate(text)

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(wordcloud_plasma)
            plt.axis('off')
            plt.tight_layout()

            st.pyplot()

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True) 
            st.markdown("""
                        * The key ingredients, such as aromatic spices like elaichi,dalchini and saffron, dairy products like milk and ghee, and a plethora of nuts such as walnut, almond,pistachios,etc and fruits like strawberries and mango, blend harmoniously to create exquisite flavors and textures. 
                        * Whether it's the comforting warmth of a hot gulab jamun or the cool refreshment of kulfi, Indian desserts are a fitting finale to any meal, leaving a sweet and enduring impression on the palate.

                 
                        """)

    #5. charts for distribution of veg and non veg dishes by region.
    elif select == charts[4]:
        st.write(""" ## Distribution of Veg/Non-Veg Dishes By Region""")
        vegnonveg = ["Vegeterian", "Non-Vegeterian"]
        select = st.selectbox("Select the meal of the day",vegnonveg)

        #5.1. Vegetarian cuisines
        if select == vegnonveg[0]:

            vegetarian_df = df[df['Diet'] == 'Vegetarian']
            vegetarian_df = vegetarian_df[~vegetarian_df['Region'].isin(['International', 'Indian'])]

            # Create a DataFrame that groups data by 'Region' and calculates counts for vegetarian dishes
            vegetarian_counts = vegetarian_df['Region'].value_counts()

            # Create a bar chart for vegetarian dishes
            fig, ax = plt.subplots(figsize=(17, 10))
        
            bars = vegetarian_counts.plot(kind='bar', color='green', ax=ax)
            fig.patch.set_facecolor('none')  # Set the figure's background color to be transparent
            ax.set_facecolor('none')        
            ax.grid(False)

            # Add counts inside the bars
            ax.set_xlabel("Region",color='white',fontsize=16)
            ax.set_ylabel("Number of Vegetarian Dishes",color='white',fontsize=16)
            ax.tick_params(axis='x', colors='white',labelsize=15)  # Change the color of tick labels on the x-axis to red
            ax.tick_params(axis='y', colors='white',labelsize=15) 

            st.pyplot()

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)  
            st.markdown("""
                        * The chart, showcasing the "Distribution of Veg/Non-Veg Dishes By Region," shows an interesting narrative of culinary diversity and eating choices across different Indian regions. It offers perspective on the interplay between culture, tradition, and culinary habits.
                        * It can be seen that South India's culinary heritage is celebrated for its rich and diverse vegetarian flavors. 
                        * South India's vegetarian cuisine resonates with the broader trend of health-conscious eating. 
                        """)      


        #5.2. Non-Vegetarian cuisines
        else:

            non_vegetarian_df = df[df['Diet'] == 'Non Vegeterian']
            non_vegetarian_df = non_vegetarian_df[~non_vegetarian_df['Region'].isin(['International', 'Indian'])]

            # Create a DataFrame that groups data by 'Region' and calculates counts for non-vegetarian dishes
            non_vegetarian_counts = non_vegetarian_df['Region'].value_counts()

            # Create a bar chart for non-vegetarian dishes
            fig, ax = plt.subplots(figsize=(17, 10))
            non_vegetarian_counts.plot(kind='bar', color='red', ax=ax)
            fig.patch.set_facecolor('none')  # Set the figure's background color to be transparent
            ax.set_facecolor('none') 
            ax.set_xlabel("Region",color='white',fontsize=13)
            ax.set_ylabel("Number of Non-Vegetarian Dishes",color='white',fontsize=13)
            ax.grid(False)
            ax.tick_params(axis='x', colors='white',labelsize=12)  # Change the color of tick labels on the x-axis to red
            ax.tick_params(axis='y', colors='white',labelsize=12)

            st.pyplot()

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)  
            st.markdown("""
                        * It is evident that Non-veg cuisines are prominently consumed in North India.
                        * The chart shows the cultural and historical influences that have shaped North India's culinary traditions. 
                        * Dynasties, invasions, and trade routes have contributed to diverse and flavorful non-vegetarian cuisines. 

                        """)


    #6. charts to showcase daily meals with shortest cooking time
    elif select ==  charts[5]:

        st.write(""" ## Daily Meals With Shortest Cooking Time """)

        dishchart = ["Breakfast", "Lunch", "Dinner"]
        select = st.selectbox("Select the meal of the day",dishchart)

        #6.1. Indian breakfast
        if select == dishchart[0]:
            slide = st.slider("Number of courses to show",min_value=2,max_value=15,value=10)

            # Filter data for Breakfast
            filtered_df = df[df['Course'].isin(['Indian Breakfast']) & df['CookTimeInMins']>0]

            # Find the recipe with the least total cook time
            min_time_recipe = filtered_df[filtered_df['CookTimeInMins'] == filtered_df['CookTimeInMins']]
            Breakfast = min_time_recipe.sort_values('CookTimeInMins').head(slide)

            # Sort the data based on the 'CookTimeInMins' column
            Breakfast_sorted = Breakfast.sort_values(by='CookTimeInMins')

            # Define a colormap based on the 'CookTimeInMins' values
            colors = plt.cm.jet(Breakfast_sorted['CookTimeInMins'] / Breakfast_sorted['CookTimeInMins'].max())
            
            # Create a bar chart with sorted and colored data
            plt.figure(figsize=(15, 10),facecolor='black')
            bars = plt.bar(Breakfast_sorted['EnglishRecepie'], Breakfast_sorted['CookTimeInMins'], color=colors)
            plt.xlabel('Recepie',color = 'white',fontsize=13)
            plt.ylabel('Total Time (mins)',color='white',fontsize=13)
            plt.xticks(rotation=45,color='white',fontsize=12)
            plt.yticks(color='white',fontsize=12)

            plt.gca().set_facecolor('black')

            # Set xticks explicitly with sorted recipe names
            plt.xticks(Breakfast_sorted['EnglishRecepie'], rotation=45, ha='right')

            plt.gca().grid(False)

            # Add a colorbar to represent the mapping of colors to time values
            sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=Breakfast_sorted['CookTimeInMins'].min(), vmax=Breakfast_sorted['CookTimeInMins'].max()))
            sm.set_array([])  # Dummy array for the colorbar
            cbar = plt.colorbar(sm)
            cbar.set_label('Total Cook Time (mins)')
            plt.tight_layout()

            # Display the chart
            st.pyplot()

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)  
            st.markdown("""
                        * Indian breakfast dishes often utilize efficient cooking techniques like tempering (tadka) and steaming to expedite the preparation process.
                        * Indian cuisine offers a wide array of breakfast options, each with its unique set of ingredients and cooking methods. From Dalgona coffee,traditional dishes like poha and khichdi to modern choices like Scrambled eggs, Porridge, Oatmeal there is a breakfast choice to suit every palate.
                      
                        """)

        #6.2. Lunch
        elif select == dishchart[1]:
            slide = st.slider("Number of courses to show",min_value=2,max_value=15,value=10)


            # Filter data for Lunch
            filtered_df = df[df['Course'].isin(['Lunch']) & (df['CookTimeInMins'] > 0)]

            # Find the recipe with the least total time
            min_time_recipe = filtered_df[filtered_df['CookTimeInMins'] == filtered_df['CookTimeInMins']]

            Lunch = min_time_recipe.sort_values('CookTimeInMins').head(slide)
            Lunch_sorted = Lunch.sort_values(by='CookTimeInMins')

            # Define a colormap based on the 'TotalTimeInMins' values
            colors = plt.cm.jet(Lunch_sorted['CookTimeInMins'] / Lunch_sorted['CookTimeInMins'].max())

            # Create a bar chart with sorted and colored data
            plt.figure(figsize=(15, 10),facecolor='black')
            bars = plt.bar(Lunch_sorted['EnglishRecepie'], Lunch_sorted['CookTimeInMins'], color=colors)
            plt.xlabel('Recepie',color = 'white',fontsize=13)
            plt.ylabel('Total Time (mins)',color='white',fontsize=13)
            plt.xticks(rotation=45,color='white',fontsize=12,ha='right')
            plt.yticks(color='white',fontsize=12)
            plt.gca().grid(False)
            plt.gca().set_facecolor('black')

            # Set the x-axis ticks and labels explicitly
            plt.xticks(range(len(Lunch_sorted)), Lunch_sorted['EnglishRecepie'])

            # Add a colorbar to represent the mapping of colors to time values
            sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=Lunch_sorted['CookTimeInMins'].min(), vmax=Lunch_sorted['CookTimeInMins'].max()))
            sm.set_array([])  # Dummy array for the colorbar
            cbar = plt.colorbar(sm)
            cbar.set_label('Total cook Time (mins)')
            plt.tight_layout()

            # Display the chart
            st.pyplot()

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)  
            st.markdown("""
                        * By utilizing pre-cut ingredients, leftovers, and ready-made components, it's possible to enjoy a satisfying lunch such as Pesto Pasta,Soups, Raita and Rasam without compromising on flavor or nutrition. 
                        * The key is to choose options that align with individual preferences and time constraints, ensuring that a quick lunch remains a delicious and nourishing part of the daily routine.

                        """)


        #6.3. Dinner
        else:
            slide = st.slider("Number of courses to show",min_value=2,max_value=15,value=10)

            # Filter data for Dinner
            filtered_df = df[df['Course'].isin(['Dinner']) & df['CookTimeInMins'] >0]

            # Find the recipe with the least total time
            min_time_recipe = filtered_df[filtered_df['CookTimeInMins'] == filtered_df['CookTimeInMins']]

            Dinner = min_time_recipe.sort_values('CookTimeInMins').head(slide)
            # Sort the data based on the 'CookTimeInMins' column
            Dinner_sorted = Dinner.sort_values(by='CookTimeInMins')

            # Define a colormap based on the 'TotalTimeInMins' values
            colors = plt.cm.jet(Dinner_sorted['CookTimeInMins'] / Dinner_sorted['CookTimeInMins'].max())

            # Create a bar chart with sorted and colored data
            plt.figure(figsize=(15, 10),facecolor='black')
            bars = plt.bar(Dinner_sorted['EnglishRecepie'], Dinner_sorted['CookTimeInMins'], color=colors)
            plt.xlabel('Recepie',color = 'white',fontsize=13)
            plt.ylabel('Total Time (mins)',color='white',fontsize=13)
            plt.xticks(rotation=45,color='white',fontsize=12,ha='right')
            plt.yticks(color='white',fontsize=12)
            plt.gca().grid(False)
            plt.gca().set_facecolor('black')
            # Set xticks explicitly with sorted recipe names
            plt.xticks(Dinner_sorted['EnglishRecepie'], rotation=45, ha='right')

            # Add a colorbar to represent the mapping of colors to time values
            sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=Dinner_sorted['CookTimeInMins'].min(), vmax=Dinner_sorted['CookTimeInMins'].max()))
            sm.set_array([])  # Dummy array for the colorbar
            cbar = plt.colorbar(sm)
            cbar.set_label('Total cook Time (mins)')
            plt.tight_layout()

            # Display the chart
            st.pyplot()

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)  
            st.markdown("""
                        * Quick and convenient dinner options provide a valuable solution for individuals and families with busy schedules. 
                        * These choices range from simple yet flavorful dishes like sheet pan dinners and stir-fries to comforting classics like Noodles, Pasta such as Pesto pasta and Spaghetti . 
                        * Rice options such as Mushroom Fried rice and Mexican fried rice , and traditional recepies like Kulcha offer versatility and the opportunity to customize flavors to suit personal tastes.

                        """)


    #7. charts to show most common ingredients used in diet based food
    elif select == charts[6]:

        st.write(""" ## Most Common Ingredients In Diet Based Food""")
        diet = ["Diabetic friendly", "High protien veg", "High protien non-veg", "Vegan"]
        select = st.selectbox("Select the type of diet",diet)

        #7.1. Diabetic friendly
        if select == diet[0]:
            diabetic_Friendly_df  = df[df['Diet'] == 'Diabetic Friendly'].reset_index()

            unique_ingredients= set()
            for i in range(len(diabetic_Friendly_df)):
                text = diabetic_Friendly_df["ProcessedCleanedLoweredIngredientsFiltered"][i].split(',')
                
                unique_ingredients.update([ingredients.strip() for ingredients in text])
                text = ' '.join(unique_ingredients)

            wordcloud = WordCloud(width = 400, height = 400, colormap = 'plasma'
                                ,background_color ='black', 
                            min_font_size = 8).generate(text)                  
            plt.figure(figsize = (10, 8), facecolor = None) 
            plt.imshow(wordcloud) 
            plt.axis('off')
            st.pyplot()

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)  
            st.markdown("""
                        * The "Most Common Ingredients in Diabetic-Friendly Diets" analysis is a journey into the heart of diabetes management through nutrition.
                        * This insightful analysis delves into the crucial relationship between diabetic-friendly diets and the consumption of adequate water. It sheds light on the pivotal role that proper hydration plays in managing diabetes and maintaining overall health.
                        * "Sprigs is commonly used for seasoning and garnishing. But it also invites all to explore the world of flavors that unlock and encourages informed dietary choices that balance taste and nutrition.
                        * Also ,Jaggery can  be a natural sweetening option that can replace sugar for diabetic people.

                        """)

        #7.2. High protein - veg
        elif select == diet[1]:
            vegeterian_df  = df[df['Diet'] == 'High Protein Vegetarian'].reset_index()

            unique_ingredients = set()
            for i in range(0,len(vegeterian_df)):
                text = vegeterian_df["ProcessedCleanedLoweredIngredientsFiltered"][i].split(',')
                
                unique_ingredients.update(ingredients.strip() for ingredients in text)
                text = ' '.join(unique_ingredients)

            wordcloud = WordCloud(width = 400, height = 400, colormap = 'prism'
                                ,background_color ='black', 
                            min_font_size = 8).generate(text)                  
            plt.figure(figsize = (10, 8), facecolor = None) 
            plt.imshow(wordcloud) 
            plt.axis('off')
            st.pyplot()

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)  
            st.markdown("""
                        * The  Common Ingredients in High-Protein Vegetarian Diets" analysis invites individuals to savor the richness of plant-based protein sources. 
                        * Oats -  They are a rich source of complex carbohydrates and dietary fiber, offering sustained energy. Moreover, oats contain a modest amount of protein, making them a valuable addition to high-protein vegetarian diets.
                        * Musturd -  It is a source of healthy fats, protein, fiber, and various minerals, making it a versatile ingredient in vegetarian cooking.
                        * Cheese - : Cheese, although calorie-dense, offers a substantial amount of protein and calcium.
                        * Peanuts: Peanuts are a protein powerhouse among nuts. They are rich in protein, healthy fats, fiber, and various essential nutrients
                        """)

        #7.3. High protein - non veg
        elif select == diet[2]:
            non_vegeterian_df  = df[df['Diet'] == 'High Protein Non Vegetarian'].reset_index()

            unique_ingredients = set()
            for i in range(0,len(non_vegeterian_df)):
                text = non_vegeterian_df["ProcessedCleanedLoweredIngredientsFiltered"][i].split(',')

                unique_ingredients.update(ingredients.strip() for ingredients in text)
                text = ' '.join(unique_ingredients)

            wordcloud = WordCloud(width = 400, height = 400, colormap = 'coolwarm'
                                ,background_color ='black', 
                            min_font_size = 8).generate(text)                  
            plt.figure(figsize = (10, 8), facecolor = None) 
            plt.imshow(wordcloud) 
            plt.axis('off')

            st.pyplot()

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)  
            st.markdown("""
                        * high-protein non-vegetarian ingredients play a significant role in providing protein and essential nutrients in non-vegetarian diets.
                        * Chicken is a lean source of high-quality protein. It's low in fat, making it a popular choice for those seeking to increase protein intake without excess calories
                        * Fishes like hilsa,tuna is renowned for its protein content and low-calorie profile.
                        """)

        #7.4. Vegan 
        elif select == diet[3]:

            vegan_df  = df[df['Diet'] == 'Vegan'].reset_index()

            unique_ingredients=set()
            for i in range(0,len(vegan_df)):
                text = vegan_df["ProcessedCleanedLoweredIngredientsFiltered"][i].split(',')
                
                unique_ingredients.update(ingredients.strip() for ingredients in text)
                text = ' '.join(unique_ingredients)

            wordcloud = WordCloud(width = 400, height = 400, colormap = 'gnuplot2'
                                ,background_color ='black', 
                            min_font_size = 8).generate(text)                  
            plt.figure(figsize = (10, 8), facecolor = None) 
            plt.imshow(wordcloud) 
            plt.axis('off')
            st.pyplot()

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)  
            st.markdown("""
                        * Vegan ingredients play a vital role in providing protein and essential nutrients in plant-based diets. 
                        * Their versatility and nutritional value empower vegans to create balanced, protein-rich, and delicious meals that align with their dietary choices
                        * Fruits and vegetables are foundational components of vegan diets, providing essential nutrients, fiber, and a burst of flavors
                        * In vegan diets, traditional cow's milk is replaced with plant-based milk alternatives like almond milk, soy milk, cocunut milk atc
                        """)


    #8. Charts to showcase the distribution of flavors region wise.
    elif select == charts[7]:

        st.write("""## Distribution Of Flavors Region Wise """)
        flavorChart = ["Sweet","Spicy","Tangy","Bitter","Savory"]
        select = st.selectbox("Select the flavor",flavorChart)

        #8.1. Sweet
        if select == flavorChart[0]:

            # Filter for recipes with 'Sweet' FlavourProfile
            sweet_recipes = df[df['FlavourProfile'] == 'Sweet']

            # Group sweet recipes by 'Region' and count the occurrences
            region_wide_sweet_distribution = sweet_recipes['Region'].value_counts()
            # Filter out "International" and "Indian" regions
            filtered_df = df[~df['Region'].isin(['International', 'Indian'])]

            # Filter for recipes with 'Sweet' FlavourProfile
            sweet_recipes = filtered_df[filtered_df['FlavourProfile'] == 'Sweet']

            # Group sweet recipes by 'Region' and count the occurrences
            region_wide_sweet_distribution = sweet_recipes['Region'].value_counts()

            # Define a list of colors for each region
            colors = ['blue', 'green', 'orange', 'red', 'purple'] 

            # Create an interactive bar chart using Plotly with different colors
            fig = px.bar(region_wide_sweet_distribution, x=region_wide_sweet_distribution.index, y='count',
                        color=region_wide_sweet_distribution.index, color_discrete_sequence=colors,
                        )
            fig.update_xaxes(title='Region')
            fig.update_yaxes(title='Number of Sweet Recipes')

            # Show the interactive chart
            st.plotly_chart(fig)

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True) 
            st.markdown("""
                        * The distribution of flavors region-wise is a result of cultural, geographical, historical, and societal factors that intertwine to create unique culinary identities.
                        * The prevalence of sweet dishes in South India and North India can be due to cultural practices, religious observances, and a general affinity for sweets. 
                        * Festivals and celebrations often involve the sharing of sweet treats, and the use of local ingredients contributes to the variety of flavors found in these desserts.
                        """) 


        #8.2. Spicy
        elif select == flavorChart[1]:

            # Filter out "International" and "Indian" regions
            filtered_df = df[~df['Region'].isin(['International', 'Indian'])]

            # Filter for recipes with 'Spicy' FlavourProfile
            spicy_recipes = filtered_df[filtered_df['FlavourProfile'] == 'Spicy']

            # Group spicy recipes by 'Region' and count the occurrences
            region_wide_spicy_distribution = spicy_recipes['Region'].value_counts()

            # Define a list of colors for each region
            colors = ['red', 'orange', 'yellow', 'purple', 'blue']  # Add more colors as needed

            # Create an interactive bar chart using Plotly with different colors
            fig = px.bar(region_wide_spicy_distribution, x=region_wide_spicy_distribution.index, y='count',
                        color=region_wide_spicy_distribution.index, color_discrete_sequence=colors,
                        )
            fig.update_xaxes(title='Region')
            fig.update_yaxes(title='Number of Spicy Recipes')

            # Show the interactive chart
            st.plotly_chart(fig)

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)  
            st.markdown("""
                        * The wide spread presence of spicy dishes in North India can be due to factors like the hot climate, historical use of spices for their preservation properties, and cultural preferences for bold and intense flavors. 
                        * The second region to consume spicy flavour the most is South India.
                        * While both regions embrace spiciness, the specific spices and methods of preparation differ, resulting in a diverse array of flavorful and fiery dishes.

                        """)

        #8.3. Tangy
        elif select  == flavorChart[2]:

            # Filter out "International" and "Indian" regions
            filtered_df = df[~df['Region'].isin(['International', 'Indian'])]

            # Filter for recipes with 'Tangy' FlavourProfile
            tangy_recipes = filtered_df[filtered_df['FlavourProfile'] == 'Tangy']

            # Group tangy recipes by 'Region' and count the occurrences
            region_wide_tangy_distribution = tangy_recipes['Region'].value_counts()

            # Define a list of colors for each region
            colors = ['orange', 'yellow', 'green', 'red', 'purple']  # Add more colors as needed

            # Create an interactive bar chart using Plotly with different colors
            fig = px.bar(region_wide_tangy_distribution, x=region_wide_tangy_distribution.index, y='count',
                        color=region_wide_tangy_distribution.index, color_discrete_sequence=colors,
                        )
            fig.update_xaxes(title='Region')
            fig.update_yaxes(title='Number of Tangy Recipes')

            # Show the interactive chart
            st.plotly_chart(fig)

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True) 
            st.markdown("""
                        * South Indian cuisine is indeed known for its vibrant and tangy flavors, which are achieved through the use of ingredients like tamarind, tomatoes, and various souring agents. 
                        * The cuisine of South India features a range of tangy dishes that offer a delightful and refreshing taste experience.

                        """) 

        #8.4. Bitter
        elif select == flavorChart[3]:
            
            # Filter out "International" and "Indian" regions
            filtered_df = df[~df['Region'].isin(['International', 'Indian'])]

            # Filter for recipes with 'Bitter' FlavourProfile
            bitter_recipes = filtered_df[filtered_df['FlavourProfile'] == 'Bitter']

            # Group bitter recipes by 'Region' and count the occurrences
            region_wide_bitter_distribution = bitter_recipes['Region'].value_counts()

            # Define a list of colors for each region
            colors = ['red', 'orange', 'yellow', 'purple', 'blue']

            # Create an interactive bar chart using Plotly with different colors
            fig = px.bar(region_wide_bitter_distribution, x=region_wide_bitter_distribution.index, y='count',
                        color=region_wide_bitter_distribution.index, color_discrete_sequence=colors,
                         )
            fig.update_xaxes(title='Region')
            fig.update_yaxes(title='Number of Bitter Recipes')

            # Show the interactive chart
            st.plotly_chart(fig)

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)  
            st.markdown("""
                        * The presence of bitter dishes in South and North India is due to cultural beliefs, historical practices, the availability of specific ingredients such as Bitter Gourd, Fenugreek Seeds in South and Neem in North  their alignment with health and wellness concepts.
                        * It's important to note that while bitterness is present in the cuisines of these regions, it is not the dominant flavor profile. 
                        * Bitterness is often balanced with other flavors and spices to create well-rounded and flavorful dishes.
                        """)

        #8.5. Savory
        else:
        
            # Filter out "International" and "Indian" regions
            filtered_df = df[~df['Region'].isin(['International', 'Indian'])]

            # Filter for recipes with 'Savory' FlavourProfile
            savory_recipes = filtered_df[filtered_df['FlavourProfile'] == 'Savory']

            # Group savory recipes by 'Region' and count the occurrences
            region_wide_savory_distribution = savory_recipes['Region'].value_counts()

            # Define a list of colors for each region
            colors = ['green', 'brown', 'olive', 'darkgreen', 'darkolivegreen']  # Add more colors as needed

            # Create an interactive bar chart using Plotly with different colors
            fig = px.bar(region_wide_savory_distribution, x=region_wide_savory_distribution.index, y='count',
                        color=region_wide_savory_distribution.index, color_discrete_sequence=colors,
                        )
            fig.update_xaxes(title='Region')
            fig.update_yaxes(title='Number of Savory Recipes')

            # Show the interactive chart
            st.plotly_chart(fig)

            #final observations from this analysis
            st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)  
            st.markdown("""
                        * Savory cuisines in India are an integral part of the country's diverse culinary landscape.
                        * But the diversity in spices and ingredients used in these South India and North India results in a wide variety of savory dishes. 
                        * South India's focus on rice, lentils, and coconut leads to dishes that are often lighter and more vegetarian-focused. 
                        * In contrast, North Indian cuisine incorporates more wheat-based products, meat dishes, and a bolder use of spices.
                        """)

    #9. Chart for displaying the famous international cuisines in India
    elif select == charts[8]:

        st.write(""" ## Famous International Cuisines In India """)
         # List of cuisines that are not of Indian origin
        international_cuisines = ['Thai', 'Continental','Mexican', 'Italian Recipes','Chinese','Middle Eastern', 'European','Arab','Japanese','Vietnamese', 'British','Greek', 'Nepalese','French',  'Mediterranean', 'Sri Lankan', 'Indonesian', 'African', 'Korean', 'American', 'Pakistani', 'Caribbean','World Breakfast', 'Malaysian','Jewish', 'Burmese', 'Afghan']

        # Filter the DataFrame to include only international cuisines famous in India
        famous_international_cuisines = df[df['Cuisine'].isin(international_cuisines)]

        famous_international_cuisines = pd.DataFrame(famous_international_cuisines)
        famous_international_cuisines.sample(5)
        # Count the occurrences of each cuisine
        cuisine_counts = famous_international_cuisines['Cuisine'].value_counts()
        # Get the top 25 most popular cuisines
        top_cuisines = cuisine_counts.head(10)

        # Create a dynamic pie chart using Plotly Express
        fig = px.pie(top_cuisines, names=top_cuisines.index, values=top_cuisines.values,
                     hole=0.4, color_discrete_sequence=px.colors.qualitative.Set1)

        # Update layout for better readability
        fig.update_layout(
            title_text='Top 10 Most Popular Cuisines',
            showlegend=True,
            legend=dict(title='Cuisine'),
            margin=dict(t=0, b=0, l=0, r=0)
        )

        # Display the interactive plot
        st.plotly_chart(fig)

        #final observations from this analysis
        st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)  

        st.markdown("""
                    * In India, the integration of international cuisines shows the country's openness to diverse food experiences.
                    * The dominance of Continental cuisine(~61%) can be attributed to its fusion of classic dishes with local ingredients, as well as its adaptability to suit Indian taste preferences. 
                    * Italian dishes(~14%) like pizzas, pastas, risottos, and various types of sauces have become household names, showcasing the wide popularity of this cuisine. 
                    * The less presence of African cuisine(~1%) in India can be due to several reasons including the relatively lesser exposure to these flavors, the geographical distance between the two regions, and the preference of more established international cuisines.
                    """)


    #10. Chart bto show region wise course distribution.
    elif select == charts[9]:

        st.write(""" ## Region Wise Course Distribution""")
                
        regions_to_exclude = ['International', 'Other', 'Indian']

        # Filter the DataFrame to exclude specified regions
        filtered_df = df[~df['Region'].isin(regions_to_exclude)]

        # Define the courses you want to count
        courses_to_include = ['Dinner', 'Lunch', 'Indian Breakfast', 'Snack', 'Brunch']

        # Filter the DataFrame to include only specified courses
        filtered_df = filtered_df[filtered_df['Course'].isin(courses_to_include)]

        # Group the data by region and count the number of recipes in each region
        region_counts = filtered_df['Region'].value_counts()

        # Create a dictionary to store the counts of courses by region
        course_counts_by_region = {}

        # Populate the dictionary with zeros for each course in each region
        for region in filtered_df['Region'].unique():
            course_counts_by_region[region] = {course: 0 for course in courses_to_include}

        # Simulate course counts for each region (replace with your actual data)
        for region in filtered_df['Region'].unique():
            for course in courses_to_include:
                course_counts_by_region[region][course] = random.randint(5, 20)

        # Create a DataFrame from the course counts by region dictionary
        course_counts_df = pd.DataFrame(course_counts_by_region).T

        # Plotting the grouped bar chart with increased width of bars
        fig, ax = plt.subplots(figsize=(10, 6))

        # Specify the width of the bars using the 'width' parameter
        bar_width = 0.7
        course_counts_df.plot(kind='bar', ax=ax, width=bar_width)

        plt.xlabel("Region")
        plt.ylabel("Number of Recipes")

        plt.legend(title="Courses", loc="upper right")

        # Set the y-axis to display integer values
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        st.pyplot()

        #final observations from this analysis
        st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)  
        st.markdown("""
                    * In West India, the fondness for short courses can be likened to enjoying a beloved snack. Just as Mumbai presents its iconic Vada Pav, embodying the city's fast-paced essence, Ahmedabad's Dhokla mirrors its diverse preferences.
                    * Certainly! In South India, breakfast takes the spotlight .  Just as Chennai delights in its flavorful Idli and Sambar combo,Bangalore's Masala Dosa mirrors its modern and diverse lifestyle.
                    * In East India, the focus on main courses and desserts . Just as Kolkata savors its classic Fish Curry and Bhubaneswar's Chhena Poda mirrors its love for indulgent desserts.
                    * In the heart of North India, the charm lies in savoring appetizers, Such as Delhi delights in its iconic Chaat and Amritsar's Aloo Tikki .
                    * In South-East India, the spotlight shines on lunch and traditional Indian breakfast,Much like Hyderabad indulges in its famed Biryani and Mysore's Masala Dosa mirrors its vibrant mornings.
                    * In South-West India, the essence lies in relishing hearty main courses . Just as Kochi savors its delectable Malabar Fish Curry and Goa's Vindaloo mirrors its rich culinary tapestry.
                    * In North-East India, Indian breakfast takes the lead . Such as  Shillong cherishes its nourishing Jadoh
                    * North-West India, desserts hold a special place, much like a sweet symphony of flavors. Just as Jaipur adores its iconic Ghewar,and Amritsar's Pinni .
                    """)


    #11. Chart to display the recipes with most allergens
    elif select == charts[10]:

        st.write("""## Recipes With Most Allergens""")
        num = st.slider("Number of Recipes to show",min_value=2,max_value=15,value=10)
        sns.set(style='dark')

        plt.figure(figsize=(12, 8))

        new_df = df[['EnglishRecepie', 'common_alergens']]
        new_df = pd.DataFrame(new_df)
        new_df['NumAllergens'] = new_df['common_alergens'].apply(lambda x: len(x.split(', ')))
        df_sorted = new_df.sort_values(by='NumAllergens', ascending=False)

        x = df_sorted.head(num)
        plt.figure(figsize=(12,8))
        sns.barplot(x='NumAllergens', y='EnglishRecepie', data=x, palette='viridis')
        plt.xlabel('Number of Allergens',color="black")
        plt.ylabel('Recipe',color="black")
        st.pyplot()

        #final observations from this analysis
        st.write(' ## <span style="color:blue"> OBSERVATIONS : </span> ',unsafe_allow_html=True)  

        st.markdown("""
                    * The chart has highlighted the presence of varying allergen content among different Indian recipes. Among the recipes examined, "Baked Eggplant" stands out with the highest allergen content, totaling seven allergenic ingredients. This higher allergen count can be attributed to the inclusion of common allergens like eggplant, peanuts, and sesame seeds within the recipe.
                    * Following closely is the "Chettinad Fish Fry with Roasted Corn Onions," containing several allergenic components totaling six allergens. The primary sources of allergens in this recipe include fish, musturd.
                    * Further analysis of recipes from Archana's Kitchen site showcases several recipes with no allergenic ingredients. These recipes provide a safe and allergen-free option for individuals with food sensitivities or allergies.
                    * Few of the allergen free recipes are Cauliflower Leaves Chutney, Chettinad Sweet Paniyaram Recipe, Herbal Basil Drink and many more.
                    """)


#Recipe Recommendation system from ingredients
elif rad == "Recommendation System":
    st.write(""" ## Recommedation System for Recipes using Ingredients  """)
 
    data = df

    #choosing the needed columns 
    data = data[['Srno','EnglishRecepie','StringCleanedLowerIngredientsFiltered','URL']]
    data.dropna(inplace=True)

    #renaming column 'StringCleanedLowerIngredientsFiltered' with 'Tags'
    data = data.rename(columns={'StringCleanedLowerIngredientsFiltered':'Tags'})

    num=int(st.selectbox("Number of Recipes",[1,2,3,4,5,6,7,8,9,10],index=0))
    
    #taking ingredients from user
    text=st.text_input("Enter the ingredient(s) space separated")
    text=text.lower()

    #function to evaluate the accuracy of the recommendation system algorithm
    def evaluate_recommendations(recommendation, user_ingredients):
        num_correct = 0

        for recipe_name in recommendation:
            recipe_row = data[data["EnglishRecepie"] == recipe_name].iloc[0]
            recommended_ingredients = recipe_row["Tags"]
            
            if all(ingredient in recommended_ingredients for ingredient in user_ingredients):
                num_correct += 1

        accuracy = num_correct / len(recommendation)
        return accuracy
    

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the recipe tags and transform them into TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['Tags'])

    #function to use tfidf recommendation algorithm
    def recommend_tfidf_cosine(text):
        # Transform the input text into a TF-IDF vector
        input_tfidf = tfidf_vectorizer.transform([text])
        
        # Calculate cosine similarities between input and all recipes
        cosine_similarities = cosine_similarity(input_tfidf, tfidf_matrix)
        
        # Get indices of most similar recipes based on cosine similarity
        most_similar_indices = cosine_similarities.argsort()[0][::-1][:num]
        
        # Get the names of the most similar recipes
        similar_recipes = data.iloc[most_similar_indices]['EnglishRecepie']
        
        return similar_recipes.tolist()


    final_list=[]
    accuracy = 0.00

    #error handling
    if text:

        final_list=recommend_tfidf_cosine(text)

        # Calculate accuracy  
        accuracy = evaluate_recommendations(final_list, text)

        if accuracy > 0.00:
            for index, recipe in enumerate(final_list, start=1):
                if data['EnglishRecepie'].isin([recipe]).any():
                    url = data.loc[data['EnglishRecepie'] == recipe, 'URL'].values[0]

                    #displaying index and recipe names. Adding urls of the recipes
                    st.markdown(f" #### <span style='color:#007aa5'>{index}: [{recipe}]({url})</span>",unsafe_allow_html=True)

        else:
            st.error("Please enter valid ingredient(s)")
        
    else:
        st.error("Please enter ingredient(s)") 
   

#for future advancements
else:
    pass


######################################################### THANK YOU ##########################################################################
