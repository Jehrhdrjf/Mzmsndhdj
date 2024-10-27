import numpy as np
import discord
import json
import os 
import time
from discord.app_commands import Choice
from discord import app_commands
import math
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from keep_alive import keep_alive

keep_alive()

current_path = os.path.dirname(os.path.abspath(__file__))
print("The directory where this script is located:", current_path)

dir = '/opt/render/project/src'




class aclient(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.synced = False

    async def on_ready(self):
        await self.wait_until_ready()
        if not self.synced:
            await tree.sync()
            self.synced = True
        print(f"We have logged in as {self.user}.")


client = aclient()
tree = app_commands.CommandTree(client)
customerrole = 'paid user'

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def prepare_data(minelocations, safespots):
    n_cells = 26
    X = np.zeros((n_cells, 9))
    y = np.zeros(n_cells)
    
    for mine in minelocations:
        y[mine] = -1
    
    for i in range(n_cells):
        if y[i] != -1:
            for idx, j in enumerate([-6, -5, -4, -1, 0, 1, 4, 5, 6]):
                if 0 <= i + j < n_cells:
                    X[i, idx] = y[i + j] != -1 
            y[i] = 1
    
    return X, y

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, activation='relu', solver='adam', random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def TrafficRetrieve(minelocations, safespots):
    X, y = prepare_data(minelocations, safespots)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, scaler = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, scaler, X_test, y_test)
    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", report)
    
    n_cells = 26
    probability = np.zeros(n_cells)
    
    for i in range(n_cells):
        features = X[i].reshape(1, -1)  
        probability[i] = model.predict_proba(scaler.transform(features))[0, 1]  
    
    tiles = min(safespots, np.sum(y != -1))  
    safe_spots_indices = np.argsort(probability)[-tiles:]  
    
    prediction = '\n'.join(''.join(['✅' if (i * 5 + j) in safe_spots_indices else '❌' for j in range(5)]) for i in range(5))
    
    return prediction



def prepare_data(minelocations_list, safespots):
    n_cells = 26
    X = np.zeros((n_cells, 9))
    y = np.zeros(n_cells) 

    for mine in minelocations_list:
        y[mine] = -1
    
    for i in range(n_cells):
        if y[i] != -1: 
            for idx, j in enumerate([-6, -5, -4, -1, 0, 1, 4, 5, 6]):
                if 0 <= i + j < n_cells:
                    X[i, idx] = y[i + j] != -1 
            y[i] = 1 

    return X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def GridDeploy(minelocations_list, safespots):
    X, y = prepare_data(minelocations_list, safespots)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    selector = SelectFromModel(clf, threshold='median')
    selector.fit(X_scaled, y)
    X_selected = selector.transform(X_scaled)
    

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    

    model = train_model(X_train, y_train)
    

    accuracy, report = evaluate_model(model, X_test, y_test)
    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", report)
    
    n_cells = 26
    probability = np.zeros(n_cells)
    
    for i in range(n_cells):
        features = X_selected[i].reshape(1, -1)  
        probability[i] = model.predict_proba(features)[0, 1]  
    
    tiles = min(safespots, np.sum(y != -1))  
    safe_spots_indices = np.argsort(probability)[-tiles:]  
    
    prediction = '\n'.join(''.join(['⭐' if (i * 5 + j) in safe_spots_indices else '❌' for j in range(5)]) for i in range(5))
    
    return prediction




import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def prepare_data(minelocations_list, safespots):
    n_cells = 26
    X = np.zeros((n_cells, 9))  
    y = np.zeros(n_cells)  

    for mine in minelocations_list:
        y[mine] = -1
    

    for i in range(n_cells):
        if y[i] != -1: 
            for idx, j in enumerate([-6, -5, -4, -1, 0, 1, 4, 5, 6]):
                if 0 <= i + j < n_cells:
                    X[i, idx] = y[i + j] != -1  
            y[i] = 1 
    
    return X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

from sklearn.svm import SVC

def SeedReversal(minelocations_list, safespots):
    X, y = prepare_data(minelocations_list, safespots)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = SVC(kernel='linear', probability=True, random_state=42) 
    selector = SelectFromModel(clf, threshold='median')
    selector.fit(X_scaled, y)
    X_selected = selector.transform(X_scaled)
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train) 
    

    accuracy, report = evaluate_model(model, X_test, y_test)
    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", report)
    
    n_cells = 26
    probability = np.zeros(n_cells)
    
    for i in range(n_cells):
        features = X_selected[i].reshape(1, -1) 
        probability[i] = model.predict_proba(features)[0, 1]
    
    tiles = min(safespots, np.sum(y != -1))
    safe_spots_indices = np.argsort(probability)[-tiles:]
    
    prediction = '\n'.join(''.join(['⭐' if (i * 5 + j) in safe_spots_indices else '❌' for j in range(5)]) for i in range(5))
    
    return prediction



def prepare_data(minelocations_list, safespots):
    n_cells = 26
    X = np.zeros((n_cells, 9))
    y = np.zeros(n_cells)
    
    for mine in minelocations_list:
        y[mine] = -1
    
    for i in range(n_cells):
        if y[i] != -1:
            for idx, j in enumerate([-6, -5, -4, -1, 0, 1, 4, 5, 6]):
                if 0 <= i + j < n_cells:
                    X[i, idx] = y[i + j] != -1 
            y[i] = 1
    
    return X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

from sklearn.ensemble import GradientBoostingClassifier

def BlazeAI(minelocations_list, safespots):
    X, y = prepare_data(minelocations_list, safespots)
    

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)  
    selector = SelectFromModel(clf, threshold='median')
    selector.fit(X_scaled, y)
    X_selected = selector.transform(X_scaled)
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train) 
    
    accuracy, report = evaluate_model(model, X_test, y_test) 
    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", report)
    
    n_cells = 26
    probability = np.zeros(n_cells)
    
    for i in range(n_cells):
        features = X_selected[i].reshape(1, -1)  
        probability[i] = model.predict_proba(features)[0, 1]  
    
    tiles = min(safespots, np.sum(y != -1))  
    safe_spots_indices = np.argsort(probability)[-tiles:]  
    
    prediction = '\n'.join(''.join(['⭐' if (i * 5 + j) in safe_spots_indices else '❌' for j in range(5)]) for i in range(5))
    
    return prediction


def prepare_data(minelocations_list, safespots):
    n_cells = 26
    X = np.zeros((n_cells, 9))  
    y = np.zeros(n_cells)  
    
    for mine in minelocations_list:
        y[mine] = -1
    
    for i in range(n_cells):
        if y[i] != -1: 
            for idx, j in enumerate([-6, -5, -4, -1, 0, 1, 4, 5, 6]):
                if 0 <= i + j < n_cells:
                    X[i, idx] = y[i + j] != -1 
            y[i] = 1
    
    return X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report











with open(f"{dir}/predmethods.json", "r") as f:
    predmethods = json.load(f)
    print(predmethods)


@app_commands.choices(prediction_method=[
    Choice(name="1. Traffic Retrieve", value="TrafficRetrieve"),
    Choice(name="2: Seed Reversal (BLAZE V2)", value="SeedReversal"),
    Choice(name="3: Grid Deploy (BLAZE V2)", value="GridDeploy"),
    Choice(name="4: Blaze AI (BLAZE V2)", value="BlazeAI"),
])
@tree.command(name='setminesmethod', description='🔮 Set Your Prediction method for growdice mines, Blaze V2.5')
async def mines(interaction: discord.Interaction, prediction_method: str):
    user = interaction.user




    if "paid v1" not in [role.name.lower() for role in user.roles]:
        error_embed = discord.Embed(
            title='❌ An Error Has Occurred.',
            description=f'{user.mention}, you have not purchased `Blaze Predictor V2.5` yet. Please purchase our predictor by opening a ticket in #ticket and check out our pricing in #pricing.',
            color=discord.Color.red()
        )
        error_embed.set_thumbnail(url='https://cdn1.iconfinder.com/data/icons/business-finance-1-1/128/buy-with-cash-1024.png')
        error_embed.set_footer(text='Error code: 494.')
        await interaction.response.send_message(embed=error_embed)
        return
    user = interaction.user
    embed = discord.Embed(title=f'> ```✔️``` Successfully set your mines prediction method, `Blaze V2.5`', description=f'We Have Sucessfully set your `prediction` method to `{prediction_method},` **{user}**', color=discord.Color.blue())
    embed.add_field(name=f'> ```👍``` {user} You can change your prediction method again!', value=f'you can change it by rerunning this command with a different `option`.')
    embed.set_thumbnail(url='https://images-ext-1.discordapp.net/external/vbPIErddi32FtDvYv4tN5Hgb8CTn0nvR3IOrfRd_jV8/https/cdn-icons-png.flaticon.com/512/1605/1605552.png?format=webp&quality=lossless&width=420&height=420')
    embed.set_footer(text=f'Thanks for using Blaze V2.5, {user}')
    if str(interaction.user.id) in predmethods:
        predmethods[str(interaction.user.id)]["minesmethod"] = prediction_method
        embed.add_field(name='> ```❗``` To Let You know:', value=f'you have already set your method for `mines` in blaze v2.5, but we changed it to `{prediction_method}`')
    else:
        predmethods[str(interaction.user.id)] = {"minesmethod": prediction_method}
    with open(f"{dir}/predmethods.json", "w") as f:
        json.dump(predmethods, f)
    await interaction.response.send_message(embed=embed)

@tree.command(name='mines', description='⭐ Predict your minesweeper game on growdice, Blaze V2.5')
async def mines(interaction: discord.Interaction, mine_locations: str, safe_spots_amount: int):
    user = interaction.user




    if "paid v1" not in [role.name.lower() for role in user.roles]:
        error_embed = discord.Embed(
            title='❌ An Error Has Occurred.',
            description=f'{user.mention}, you have not purchased `Blaze Predictor V2.5` yet. Please purchase our predictor by opening a ticket in #ticket and check out our pricing in #pricing.',
            color=discord.Color.red()
        )
        error_embed.set_thumbnail(url='https://cdn1.iconfinder.com/data/icons/business-finance-1-1/128/buy-with-cash-1024.png')
        error_embed.set_footer(text='Error code: 494.')
        await interaction.response.send_message(embed=error_embed)
        return
    user = interaction.user
    embedgen = discord.Embed(title='> ```🔥``` We are generating your Mines prediction with `Blaze.`', description='Please be patient as this proccess will take 2-3 seconds..', color=discord.Color.orange())
    embedgen.set_image(url='https://media.discordapp.net/attachments/1161352078636105738/1229044253884284979/standard_10.gif?ex=662e3fa8&is=661bcaa8&hm=c8fad34e84c67cd74526464b9ad91c65b86dd974d92500aaecb7b54ebc6c7acb&=&width=747&height=420')
    await interaction.response.send_message(embed=embedgen)
    time.sleep(2)

    individual = [int(x.strip()) for x in mine_locations.split(',')]
    if any(num > 25 for num in individual):
        embed3 = discord.Embed(title=f'> 🟡 {user}, Some values in your `mine_locations` list are greater than 25.', description='Please make sure you provide correct numbers so you don\'t break our `Database.`', color=discord.Color.yellow())
        embed3.set_footer(text='Blaze V2.5')
        await interaction.edit_original_response(embed=embed3)
        return
    mine_locations2 = list(map(int, mine_locations.split(',')))


    methods = predmethods.get(str(interaction.user.id))
    if methods:
        method = methods.get("minesmethod")
    else:
        Ot = discord.Embed(title='> ```❌``` We were unable to detect your ID in our Database.', description='Please set your prediction `method` to procceed.', color=discord.Color.red())
        Ot.set_footer(text='blaze v2.5')
        await interaction.edit_original_response(embed=Ot)
        return
    if method:
        method = method
    else:
        emberror = discord.Embed(title='> ```🟣``` You have not set your prediction method `yet.`', description='Please do it to continue.', color=discord.Color.purple())
        emberror.set_footer(text='Blaze v2.5')
        await interaction.edit_original_response(embed=emberror)
        return
    minelocations_list = mine_locations2
    safespots = safe_spots_amount
    prediction_method = method
    if prediction_method == "TrafficRetrieve":
        prediction = TrafficRetrieve(mine_locations2, safe_spots_amount)
    elif prediction_method == "SeedReversal":
        prediction = SeedReversal(minelocations_list, safespots)
    elif prediction_method == "GridDeploy":
        prediction = GridDeploy(minelocations_list, safespots)
    elif prediction_method == "BlazeAI":
        prediction = BlazeAI(minelocations_list, safespots)
    elif prediction_method == "VioletMarathon":
        embed = discord.Embed(title=f'> ```❌``` {user}, Your `prediction` Method Is Yet Unreleased. ', description='If You Want To Use VioletMarathon, you\'ll Need to wait for notris to release it!', color=discord.Color.red())
        embed.set_footer(text='Blaze V2.5')
        await interaction.edit_original_response(embed=embed)
        return



    embed2 = discord.Embed(title='> ```⭐``` Blaze Predictor `Mines`', description='Thanks for choosing the `Best` growdice predictor, Blaze V2.5.', color=discord.Color.orange())
    embed2.add_field(name='> `💎` Sucessful Mines Prediction:', value=f'{prediction}', inline=False)
    embed2.add_field(name='> `🌱` Provided MineLocations:', value=f'`{mine_locations}`')
    embed2.add_field(name='> `🍀` Safe Spots Amount:', value=f'`{safe_spots_amount}`')
    embed2.add_field(name='> `🗣️` Predicting For:', value=f'`{user}`')
    embed2.add_field(name='> `👑` Owner:', value=f'`Haladas`', inline=False)
    embed2.add_field(name='> `👌` Probability:', value=f'`4 dollars`', inline=False)
    embed2.add_field(name='> `🔥` Prediction Method:', value=f'`{prediction_method}`', inline=False)
    embed2.add_field(name='> `🖥️` Developer:', value=f'`notris`', inline=False)
    embed2.add_field(name='> `❓` Feeling like growdice is trashing you?:', value=f'Womp womp. shit happens! we are introducing' + '\n' + 'a new command, `unrig`. it will assist you' + '\n' + 'In Losing Less Games By Tampering With Your Provably `Fairness.`', inline=False)
    embed2.add_field(name='> `❓` Issue Occured?:', value=f'We are super sorry this occured. Please ping `@notris` or `@haldas` to seek assistance.', inline=False)
    embed2.set_footer(text='Blaze V2.5')
    await interaction.edit_original_response(embed=embed2)
    return mine_locations2, safe_spots_amount, interaction


from collections import defaultdict

def TaskAssign(color1, color2, color3, color4):
    color_count = defaultdict(int)
    total_colors = 0
    pairwise_count = defaultdict(int)
    total_pairs = 0
    colors = [color1, color2, color3, color4]
    for i in range(len(colors)):
        color_count[colors[i]] += 1
        total_colors += 1
        for j in range(i+1, len(colors)):
            pair = tuple(sorted([colors[i], colors[j]]))
            pairwise_count[pair] += 1
            total_pairs += 1

    probabilities = {}
    for color, count in color_count.items():
        color_prob = count / total_colors
        pair_prob = 0
        for pair, pair_count in pairwise_count.items():
            if color in pair:
                pair_prob += pair_count / total_pairs
        probabilities[color] = (color_prob, pair_prob)

    final_probabilities = {}
    for color, (color_prob, pair_prob) in probabilities.items():
        final_probabilities[color] = color_prob + pair_prob
    max_prob = max(final_probabilities.values())
    max_prob_colors = [color for color, prob in final_probabilities.items() if prob == max_prob]
    if len(max_prob_colors) == 1:
        prediction = max_prob_colors[0]
    else:
        prediction = "Suspicious Games, I won't predict."

    return prediction





from collections import Counter

def CounterAttract(color1, color2, color3):
    outcome1 = color1
    outcome2 = color2
    outcome3 = color3
    outcomes = [outcome1, outcome2, outcome3]
    sequence_count = Counter(zip(outcomes, outcomes[1:]))
    outcome_count = Counter(outcomes)
    probabilities = {}
    for outcome in ['silver', 'gold']:
        previous_outcome = outcomes[-1]
        total_count = sum(sequence_count.values())
        sequence_probability = sequence_count.get((previous_outcome, outcome), 0) / total_count
        outcome_probability = outcome_count.get(outcome, 0) / len(outcomes)
        probabilities[outcome] = (sequence_probability + outcome_probability) / 2
    next_prediction = max(probabilities, key=probabilities.get)



    prediction = next_prediction
    return prediction



@app_commands.choices(prediction_method=[
    Choice(name="1. TaskAssign", value="TaskAssign"),
    Choice(name="2. CounterAttract (BLAZE V2)", value="CounterAttract")

])
@tree.command(name='setcoinflipmethod', description='🪙 Set Your Prediction method for growdice coinflip, Blaze V2.5')
async def coinflip(interaction: discord.Interaction, prediction_method: str):
    user = interaction.user




    if "paid v1" not in [role.name.lower() for role in user.roles]:
        error_embed = discord.Embed(
            title='❌ An Error Has Occurred.',
            description=f'{user.mention}, you have not purchased `Blaze Predictor V2.5` yet. Please purchase our predictor by opening a ticket in #ticket and check out our pricing in #pricing.',
            color=discord.Color.red()
        )
        error_embed.set_thumbnail(url='https://cdn1.iconfinder.com/data/icons/business-finance-1-1/128/buy-with-cash-1024.png')
        error_embed.set_footer(text='Error code: 494.')
        await interaction.response.send_message(embed=error_embed)
        return
    user = interaction.user
    embed = discord.Embed(title=f'> ```✔️``` Successfully set your coinflip prediction method, `Blaze V2.5`', description=f'We Have Sucessfully set your `prediction` method to `{prediction_method},` **{user}**', color=discord.Color.blue())
    embed.add_field(name=f'> ```👍``` {user} You can change your prediction method again!', value=f'you can change it by rerunning this command with a different `option`.')
    embed.set_thumbnail(url='https://images-ext-1.discordapp.net/external/vbPIErddi32FtDvYv4tN5Hgb8CTn0nvR3IOrfRd_jV8/https/cdn-icons-png.flaticon.com/512/1605/1605552.png?format=webp&quality=lossless&width=420&height=420')
    embed.set_footer(text=f'Thanks for using Blaze V2.5, {user}')
    if str(interaction.user.id) in predmethods:
        predmethods[str(interaction.user.id)]["coinflipmethod"] = prediction_method
        embed.add_field(name='> ```❗``` To Let You know:', value=f'you have already set your method for `coinflip` in blaze v2.5, but we changed it to `{prediction_method}`')
    else:
        predmethods[str(interaction.user.id)] = {"coinflipmethod": prediction_method}
    with open(f"{dir}/predmethods.json", "w") as f:
        json.dump(predmethods, f)
    await interaction.response.send_message(embed=embed)

@tree.command(name='coinflip', description = '🪙 Predict your coinflip game using the best growdice predictor, blaze v2,5')
async def thing(interaction: discord.Interaction, color1: str, color2: str, color3: str, color4: str):
    user = interaction.user




    if "paid v1" not in [role.name.lower() for role in user.roles]:
        error_embed = discord.Embed(
            title='❌ An Error Has Occurred.',
            description=f'{user.mention}, you have not purchased `Blaze Predictor V2.5` yet. Please purchase our predictor by opening a ticket in #ticket and check out our pricing in #pricing.',
            color=discord.Color.red()
        )
        error_embed.set_thumbnail(url='https://cdn1.iconfinder.com/data/icons/business-finance-1-1/128/buy-with-cash-1024.png')
        error_embed.set_footer(text='Error code: 494.')
        await interaction.response.send_message(embed=error_embed)
        return
    if color1 == 'S':
        color1 = 'silver'
    elif color1 == 'Si':
        color1 = 'silver'
    elif color1 == 'Sil':
        color1 = 'silver'
    elif color1 == 'Silv':
        color1 = 'silver'
    elif color1 == 'Silver':
        color1 = 'silver'
    elif color1 == 's':
        color1 = 'silver'

    if color2 == 'Si':
        color2 = 'silver'
    elif color2 == 'Sil':
        color2 = 'silver'
    elif color2 == 'Silv':
        color2 = 'silver'
    elif color2 == 'Silver':
        color2 = 'silver'
    elif color2 == 's':
        color2 = 'silver'

    if color3 == 'Si':
        color3 = 'silver'
    elif color3 == 'Sil':
        color3 = 'silver'
    elif color3 == 'Silv':
        color3 = 'silver'
    elif color3 == 'Silver':
        color3 = 'silver'
    elif color3 == 's':
        color3 = 'silver'

    if color4 == 'Si':
        color4 = 'silver'
    elif color4 == 'Sil':
        color4 = 'silver'
    elif color4 == 'Silv':
        color4 = 'silver'
    elif color4 == 'Silver':
        color4 = 'silver'
    elif color4 == 's':
        color4 = 'silver'

    if color1 == 'G':
        color1 = 'gold'
    elif color1 == 'Go':
        color1 = 'gold'
    elif color1 == 'Gol':
        color1 = 'gold'
    elif color1 == 'Gold':
        color1 = 'gold'
    elif color1 == 'g':
        color1 = 'gold'

    if color2 == 'Go':
        color2 = 'gold'
    elif color2 == 'Gol':
        color2 = 'gold'
    elif color2 == 'Gold':
        color2 = 'gold'
    elif color2 == 'g':
        color2 = 'gold'

    if color3 == 'Go':
        color3 = 'gold'
    elif color3 == 'Gol':
        color3 = 'gold'
    elif color3 == 'Gold':
        color3 = 'gold'
    elif color3 == 'g':
        color3 = 'gold'

    if color4 == 'Go':
        color4 = 'gold'
    elif color4 == 'Gol':
        color4 = 'gold'
    elif color4 == 'Gold':
        color4 = 'gold'
    elif color4 == 'g':
        color4 = 'gold'
    user = interaction.user
    embedgen = discord.Embed(title='> ```🔥``` We are generating your Coinflip prediction with `Blaze.`', description='Please be patient as this proccess will take 2-3 seconds..', color=discord.Color.orange())
    embedgen.set_image(url='https://media.discordapp.net/attachments/1161352078636105738/1229044253884284979/standard_10.gif?ex=662e3fa8&is=661bcaa8&hm=c8fad34e84c67cd74526464b9ad91c65b86dd974d92500aaecb7b54ebc6c7acb&=&width=747&height=420')
    await interaction.response.send_message(embed=embedgen)
    time.sleep(2)
    methods = predmethods.get(str(interaction.user.id))
    if methods:
        method = methods.get("coinflipmethod")
    else:
        Ot = discord.Embed(title='> ```❌``` We were unable to detect your ID in our Database.', description='Please set your prediction `method` to procceed.', color=discord.Color.red())
        Ot.set_footer(text='blaze v2.5')
        await interaction.edit_original_response(embed=Ot)
        return
    if method:
        method = method
    else:
        emberror = discord.Embed(title='> ```🟣``` You have not set your prediction method `yet.`', description='Please do it to continue.', color=discord.Color.purple())
        emberror.set_footer(text='Blaze v2.5')
        await interaction.edit_original_response(embed=emberror)
        return
    
    prediction_method = method
    if prediction_method == "TaskAssign":
        prediction = TaskAssign(color1, color2, color3, color4)
    elif prediction_method == "CounterAttract":
        prediction = CounterAttract(color1, color2, color3)


    if prediction == 'silver':
        prediction = 'Silver 🥈'
        colour = discord.Color.light_gray()
    elif prediction == 'gold':
        prediction = 'Gold 🥇'
        colour = discord.Color.yellow()
    elif prediction == 'Suspicious Games, i wont Predict.':
        colour = discord.Color.green()

    embed2 = discord.Embed(title='> ```⭐``` Blaze Predictor `Coinflip`', description='Thanks for choosing the Best growdice predictor, `Blaze V2.5.`', color=colour)
    embed2.add_field(name='> `🪙` Sucessful Coinflip Prediction:', value=f'`{prediction}`', inline=False)
    embed2.add_field(name='> `🌱` Provided Colors:', value=f'`{color1}, {color2}, {color3}, {color4}`')
    embed2.add_field(name='> `🗣️` Predicting For:', value=f'`{user}`')
    embed2.add_field(name='> `👑` Owner:', value=f'`Haladas`', inline=False)
    embed2.add_field(name='> `👌` Probability:', value=f'`4 dollars`', inline=False)
    embed2.add_field(name='> `🔥` Prediction Method:', value=f'`{prediction_method}`', inline=False)
    embed2.add_field(name='> `🖥️` Developer:', value=f'`notris`', inline=False)
    embed2.set_footer(text='Blaze V2.5')
    await interaction.edit_original_response(embed=embed2)


    return color1, color2, color3, color4





@app_commands.choices(prediction_method=[
    Choice(name="1. SimpleDecision.", value="SimpleDecision"),
    Choice(name="1. DrewnColour (BLAZE V2)", value="DrewnColour")

])
@tree.command(name='setslidemethod', description='🛝 Set Your Prediction method for growdice slide, Blaze V2.5')
async def slide(interaction: discord.Interaction, prediction_method: str):
    user = interaction.user




    if "paid v1" not in [role.name.lower() for role in user.roles]:
        error_embed = discord.Embed(
            title='❌ An Error Has Occurred.',
            description=f'{user.mention}, you have not purchased `Blaze Predictor V2.5` yet. Please purchase our predictor by opening a ticket in #ticket and check out our pricing in #pricing.',
            color=discord.Color.red()
        )
        error_embed.set_thumbnail(url='https://cdn1.iconfinder.com/data/icons/business-finance-1-1/128/buy-with-cash-1024.png')
        error_embed.set_footer(text='Error code: 494.')
        await interaction.response.send_message(embed=error_embed)
        return
    user = interaction.user
    embed = discord.Embed(title=f'> ```✔️``` Successfully set your slide prediction method, `Blaze V2.5`', description=f'We Have Sucessfully set your `prediction` method to `{prediction_method},` **{user}**', color=discord.Color.blue())
    embed.add_field(name=f'> ```👍``` {user} You can change your prediction method again!', value=f'you can change it by rerunning this command with a different `option`.')
    embed.set_thumbnail(url='https://images-ext-1.discordapp.net/external/vbPIErddi32FtDvYv4tN5Hgb8CTn0nvR3IOrfRd_jV8/https/cdn-icons-png.flaticon.com/512/1605/1605552.png?format=webp&quality=lossless&width=420&height=420')
    embed.set_footer(text=f'Thanks for using Blaze V2.5, {user}')
    if str(interaction.user.id) in predmethods:
        predmethods[str(interaction.user.id)]["slidemethod"] = prediction_method
        embed.add_field(name='> ```❗``` To Let You know:', value=f'you have already set your method for `slide` in blaze v2.5, but we changed it to `{prediction_method}`')
    else:
        predmethods[str(interaction.user.id)] = {"slidemethod": prediction_method}
    with open(f"{dir}/predmethods.json", "w") as f:
        json.dump(predmethods, f)
    await interaction.response.send_message(embed=embed)



def SimpleDecision(color1, color2, color3):
    colors = {'green': 3, 'red': 2, 'black': 1}
    numerical_values = [colors[color1], colors[color2], colors[color3]]
    prediction = sum(numerical_values) % 3
    predictions_map = {0: 'green', 1: 'red', 2: 'black'}
    predicted_color = predictions_map[prediction]

    return predicted_color


class RoulettePredictor:
    def __init__(self):
        self.transitions = {'red': {'red': 0, 'black': 0, 'green': 0},
                            'black': {'red': 0, 'black': 0, 'green': 0},
                            'green': {'red': 0, 'black': 0, 'green': 0}}
        self.total_counts = {'red': 0, 'black': 0, 'green': 0}

    def update_transitions(self, color1, color2, color3):
        colors = [color1.lower(), color2.lower(), color3.lower()]
        for i in range(1, len(colors)):
            self.transitions[colors[i - 1]][colors[i]] += 1
            self.total_counts[colors[i - 1]] += 1

    def predict_next_color(self, color1, color2, color3):
        self.update_transitions(color1, color2, color3)
        probabilities = self.get_transition_probabilities(color1.lower(), color2.lower(), color3.lower())
        next_color = max(probabilities, key=probabilities.get)
        return next_color

    def get_transition_probabilities(self, color1, color2, color3):
        total_count = self.total_counts[color1]
        if total_count == 0:
            return {'red': 1/3, 'black': 1/3, 'green': 1/3} 

        probabilities = {}
        for color in self.transitions[color1]:
            probabilities[color] = self.transitions[color1][color] / total_count
        return probabilities






@tree.command(name='slide', description='🎲 Predict your slide game using the best growdice predictor, blaze v2,5')
async def thing(interaction: discord.Interaction, color1: str, color2: str, color3: str):
    user = interaction.user




    if "paid v1" not in [role.name.lower() for role in user.roles]:
        error_embed = discord.Embed(
            title='❌ An Error Has Occurred.',
            description=f'{user.mention}, you have not purchased `Blaze Predictor V2.5` yet. Please purchase our predictor by opening a ticket in #ticket and check out our pricing in #pricing.',
            color=discord.Color.red()
        )
        error_embed.set_thumbnail(url='https://cdn1.iconfinder.com/data/icons/business-finance-1-1/128/buy-with-cash-1024.png')
        error_embed.set_footer(text='Error code: 494.')
        await interaction.response.send_message(embed=error_embed)
        return

    if color1 == 'R':
        color1 = 'red'
    elif color1 == 'Re':
        color1 = 'red'
    elif color1 == 'Red':
        color1 = 'red'
    elif color1 == 'r':
        color1 = 'red'

    if color2 == 'Re':
        color2 = 'red'
    elif color2 == 'Red':
        color2 = 'red'
    elif color2 == 'r':
        color2 = 'red'

    if color3 == 'Re':
        color3 = 'red'
    elif color3 == 'Red':
        color3 = 'red'
    elif color3 == 'r':
        color3 = 'red'

    if color1 == 'B':
        color1 = 'black'
    elif color1 == 'Bl':
        color1 = 'black'
    elif color1 == 'Blk':
        color1 = 'black'
    elif color1 == 'Black':
        color1 = 'black'
    elif color1 == 'b':
        color1 = 'black'

    if color2 == 'Bl':
        color2 = 'black'
    elif color2 == 'Blk':
        color2 = 'black'
    elif color2 == 'Black':
        color2 = 'black'
    elif color2 == 'b':
        color2 = 'black'

    if color3 == 'Bl':
        color3 = 'black'
    elif color3 == 'Blk':
        color3 = 'black'
    elif color3 == 'Black':
        color3 = 'black'
    elif color3 == 'b':
        color3 = 'black'

    if color1 == 'G':
        color1 = 'green'
    elif color1 == 'Gr':
        color1 = 'green'
    elif color1 == 'Grn':
        color1 = 'green'
    elif color1 == 'Green':
        color1 = 'green'
    elif color1 == 'g':
        color1 = 'green'

    if color2 == 'Gr':
        color2 = 'green'
    elif color2 == 'Grn':
        color2 = 'green'
    elif color2 == 'Green':
        color2 = 'green'
    elif color2 == 'g':
        color2 = 'green'

    if color3 == 'Gr':
        color3 = 'green'
    elif color3 == 'Grn':
        color3 = 'green'
    elif color3 == 'Green':
        color3 = 'green'
    elif color3 == 'g':
        color3 = 'green'

    if color3 == 'Re':
        color3 = 'red'


    user = interaction.user
    embedgen = discord.Embed(title='> ```🔥``` We are generating your Slide prediction with `Blaze.`', description='Please be patient as this proccess will take 2-3 seconds..', color=discord.Color.orange())
    embedgen.set_image(url='https://media.discordapp.net/attachments/1161352078636105738/1229044253884284979/standard_10.gif?ex=662e3fa8&is=661bcaa8&hm=c8fad34e84c67cd74526464b9ad91c65b86dd974d92500aaecb7b54ebc6c7acb&=&width=747&height=420')
    await interaction.response.send_message(embed=embedgen)
    time.sleep(2)
    methods = predmethods.get(str(interaction.user.id))
    if methods:
        method = methods.get("slidemethod")
    else:
        Ot = discord.Embed(title='> ```❌``` We were unable to detect your ID in our Database.', description='Please set your prediction `method` to procceed.', color=discord.Color.red())
        Ot.set_footer(text='blaze v2.5')
        await interaction.edit_original_response(embed=Ot)
        return
    if method:
        method = method
    else:
        emberror = discord.Embed(title='> ```🟣``` You have not set your prediction method `yet.`', description='Please do it to continue.', color=discord.Color.purple())
        emberror.set_footer(text='Blaze v2.5')
        await interaction.edit_original_response(embed=emberror)
        return
    
    prediction_method = method
    if prediction_method == "SimpleDecision":
        prediction = SimpleDecision(color1, color2, color3)
    elif prediction_method == "DrewnColour":
        predictor = RoulettePredictor()
        prediction = predictor.predict_next_color(color1, color2, color3)


    if prediction == 'black':
        prediction = 'Black ⚫'
        colour = discord.Color.default()
    elif prediction == 'red':
        prediction = 'Red 🔴'
        colour = discord.Color.red()
    elif prediction == 'green':
        prediction = 'Green 🟢'
        colour = discord.Color.green()


    embed2 = discord.Embed(title='> ```⭐``` Blaze Predictor `Slide`', description='Thanks for choosing the Best growdice predictor, `Blaze V2.5.`', color=colour)
    embed2.add_field(name='> `🎲` Sucessful Slide Prediction:', value=f'`{prediction}`', inline=False)
    embed2.add_field(name='> `🌱` Provided Colors:', value=f'`{color1}, {color2}, {color3}`')
    embed2.add_field(name='> `🗣️` Predicting For:', value=f'`{user}`')
    embed2.add_field(name='> `👑` Owner:', value=f'`Haladas`', inline=False)
    embed2.add_field(name='> `👌` Probability:', value=f'`4 dollars`', inline=False)
    embed2.add_field(name='> `🔥` Prediction Method:', value=f'`{prediction_method}`', inline=False)
    embed2.add_field(name='> `🖥️` Developer:', value=f'`notris`', inline=False)
    embed2.set_footer(text='Blaze V2.5')
    await interaction.edit_original_response(embed=embed2)


    return color1, color2, color3




@app_commands.choices(prediction_method=[
    Choice(name="1. SimpleAverage.", value="SimpleAverage"),
    Choice(name="2. NumberTrend.", value="NumberTrend"),
    Choice(name="3. Define. (BLAZE V2)", value="Define"),
    Choice(name="4. PathDefend. (BLAZE V2)", value="PathDefend"),
    Choice(name="5. AverageOutcome. (BLAZE V2)", value="AverageOutcome"),
    Choice(name="6. PathPredictingV4. (BLAZE V2)", value="PathPredictingV4")

])
@tree.command(name='setcrashmethod', description='💥 Set Your Prediction method for growdice crash, Blaze V2.5')
async def crash(interaction: discord.Interaction, prediction_method: str):
    user = interaction.user




    if "paid v1" not in [role.name.lower() for role in user.roles]:
        error_embed = discord.Embed(
            title='❌ An Error Has Occurred.',
            description=f'{user.mention}, you have not purchased `Blaze Predictor V2.5` yet. Please purchase our predictor by opening a ticket in #ticket and check out our pricing in #pricing.',
            color=discord.Color.red()
        )
        error_embed.set_thumbnail(url='https://cdn1.iconfinder.com/data/icons/business-finance-1-1/128/buy-with-cash-1024.png')
        error_embed.set_footer(text='Error code: 494.')
        await interaction.response.send_message(embed=error_embed)
        return
    user = interaction.user
    embed = discord.Embed(title=f'> ```✔️``` Successfully set your crash prediction method, `Blaze V2.5`', description=f'We Have Sucessfully set your `prediction` method to `{prediction_method},` **{user}**', color=discord.Color.blue())
    embed.add_field(name=f'> ```👍``` {user} You can change your prediction method again!', value=f'you can change it by rerunning this command with a different `option`.')
    embed.set_thumbnail(url='https://images-ext-1.discordapp.net/external/vbPIErddi32FtDvYv4tN5Hgb8CTn0nvR3IOrfRd_jV8/https/cdn-icons-png.flaticon.com/512/1605/1605552.png?format=webp&quality=lossless&width=420&height=420')
    embed.set_footer(text=f'Thanks for using Blaze V2.5, {user}')
    if str(interaction.user.id) in predmethods:
        predmethods[str(interaction.user.id)]["crashmethod"] = prediction_method
        embed.add_field(name='> ```❗``` To Let You know:', value=f'you have already set your method for `crash` in blaze v2.5, but we changed it to `{prediction_method}`')
    else:
        predmethods[str(interaction.user.id)] = {"crashmethod": prediction_method}
    with open(f"{dir}/predmethods.json", "w") as f:
        json.dump(predmethods, f)
    await interaction.response.send_message(embed=embed)




def SimpleAverage(crashpoint1, crashpoint2, crashpoint3):
    cp1 = crashpoint1
    cp2 = crashpoint2
    cp3 = crashpoint3
    averagepred = (cp1 + cp2 + cp3) / 3
    return averagepred


def NumberTrend(crashpoint1, crashpoint2, crashpoint3):
    log_cp1 = math.log(crashpoint1)
    log_cp2 = math.log(crashpoint2)
    log_cp3 = math.log(crashpoint3)
    weight1 = log_cp1 / (log_cp1 + log_cp2 + log_cp3)
    weight2 = log_cp2 / (log_cp1 + log_cp2 + log_cp3)
    weight3 = log_cp3 / (log_cp1 + log_cp2 + log_cp3)
    weighted_avg = (weight1 * crashpoint1 + weight2 * crashpoint2 + weight3 * crashpoint3)
    
    return weighted_avg

from sklearn.neural_network import MLPRegressor
import numpy as np

def PathPredictingV4(lastcrashpoint, lastcrashpoint2):
    model = MLPRegressor(hidden_layer_sizes=(64, 64, 32), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=100)

    # notris note: nigga
    X_train = np.array([[1, lastcrashpoint], [2, lastcrashpoint2]])
    y_train = np.array([lastcrashpoint, lastcrashpoint2])


    model.fit(X_train, y_train)

    prediction = model.predict([[2, lastcrashpoint2]])

    return prediction[0]

import numpy as np
from scipy.optimize import curve_fit

import numpy as np
from scipy.optimize import curve_fit


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DefineModel(nn.Module):
    def __init__(self):
        super(DefineModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def Define(lastcrashpoint, lastcrashpoint2):
    model = DefineModel()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train = torch.tensor([[1, lastcrashpoint], [2, lastcrashpoint2]], dtype=torch.float32)
    y_train = torch.tensor([[lastcrashpoint], [lastcrashpoint2]], dtype=torch.float32)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    prediction = model(torch.tensor([[2, lastcrashpoint2]], dtype=torch.float32)).item()

    return prediction




import tensorflow as tf

def PathDefend(lastcrashpoint, lastcrashpoint2):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')

    X_train = tf.constant([[1, lastcrashpoint], [2, lastcrashpoint2]], dtype=tf.float32)
    y_train = tf.constant([[lastcrashpoint], [lastcrashpoint2]], dtype=tf.float32)
    model.fit(X_train, y_train, epochs=100, verbose=0, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

    prediction = model.predict([[2, lastcrashpoint2]])[0][0]


    return prediction


def AverageOutcome(lastcrashpoint, lastcrashpoint2):
    average = sum([lastcrashpoint, lastcrashpoint2]) / 2  
    prediction = (1 / (average - 2) / 1)
    prediction = abs(prediction)
    if 0.01 <= prediction < 1.0:
        prediction += 1
    return prediction


@tree.command(name='crash', description='🚀 Predict your crash game using the best growdice predictor, blaze v2,5')
async def thing(interaction: discord.Interaction, crashpoint1: float, crashpoint2: float, crashpoint3: float):
    user = interaction.user




    if "paid v1" not in [role.name.lower() for role in user.roles]:
        error_embed = discord.Embed(
            title='❌ An Error Has Occurred.',
            description=f'{user.mention}, you have not purchased `Blaze Predictor V2.5` yet. Please purchase our predictor by opening a ticket in #ticket and check out our pricing in #pricing.',
            color=discord.Color.red()
        )
        error_embed.set_thumbnail(url='https://cdn1.iconfinder.com/data/icons/business-finance-1-1/128/buy-with-cash-1024.png')
        error_embed.set_footer(text='Error code: 494.')
        await interaction.response.send_message(embed=error_embed)
        return
    user = interaction.user
    embedgen = discord.Embed(title='> ```🔥``` We are generating your Crash prediction with `Blaze.`', description='Please be patient as this proccess will take 2-3 seconds..', color=discord.Color.orange())
    embedgen.set_image(url='https://media.discordapp.net/attachments/1161352078636105738/1229044253884284979/standard_10.gif?ex=662e3fa8&is=661bcaa8&hm=c8fad34e84c67cd74526464b9ad91c65b86dd974d92500aaecb7b54ebc6c7acb&=&width=747&height=420')
    await interaction.response.send_message(embed=embedgen)
    time.sleep(2)
    methods = predmethods.get(str(interaction.user.id))
    if methods:
        method = methods.get("crashmethod")
    else:
        Ot = discord.Embed(title='> ```❌``` We were unable to detect your ID in our Database.', description='Please set your prediction `method` to procceed.', color=discord.Color.red())
        Ot.set_footer(text='blaze v2.5')
        await interaction.edit_original_response(embed=Ot)
        return
    if method:
        method = method
    else:
        emberror = discord.Embed(title='> ```🟣``` You have not set your prediction method `yet.`', description='Please do it to continue.', color=discord.Color.purple())
        emberror.set_footer(text='Blaze v2.5')
        await interaction.edit_original_response(embed=emberror)
        return
    
    prediction_method = method
    if prediction_method == "SimpleAverage":
        prediction = SimpleAverage(crashpoint1, crashpoint2, crashpoint3)
    elif prediction_method == "NumberTrend":
        prediction = NumberTrend(crashpoint1, crashpoint2, crashpoint3)
    elif prediction_method == "Define":
        prediction = NumberTrend(crashpoint1, crashpoint2)
    elif prediction_method == "PathDefend":
        prediction = PathDefend(crashpoint1, crashpoint2)
    elif prediction_method == "AverageOutcome":
        prediction = AverageOutcome(crashpoint1, crashpoint2)
    elif prediction_method == "PathPredictingV4":
        prediction = PathPredictingV4(crashpoint1, crashpoint2)


    

    embed2 = discord.Embed(title='> ```⭐``` Blaze Predictor `Crash`', description='Thanks for choosing the Best growdice predictor, `Blaze V2.5.`', color=discord.Color.blue())
    embed2.add_field(name='> `🚀` Sucessful Crash Prediction:', value=f'`{prediction}`', inline=False)
    embed2.add_field(name='> `🌱` Provided CrashPoints:', value=f'`{crashpoint1}, {crashpoint2}, {crashpoint3}`')
    embed2.add_field(name='> `🗣️` Predicting For:', value=f'`{user}`')
    embed2.add_field(name='> `👑` Owner:', value=f'`Haladas`', inline=False)
    embed2.add_field(name='> `👌` Risk Level:', value=f'`4 dollars`', inline=False)
    embed2.add_field(name='> `🔥` Prediction Method:', value=f'`{prediction_method}`', inline=False)
    embed2.add_field(name='> `🖥️` Developer:', value=f'`notris`', inline=False)
    embed2.set_footer(text='Blaze V2.5')
    await interaction.edit_original_response(embed=embed2)


    return crashpoint1, crashpoint2, crashpoint3





client.run("MTMwMDA2MTI2MDc0MTg2OTU3OQ.G7qAYO.gnXOAgCV4mRYK0bK3cbBkd9NTrB_fjdwvqbZww")
