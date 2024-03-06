from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Primeiro modelo

# Aqui estamos carregando o modelo já treinado que está no arquivo JobLib
with open('C:/Users/HP/Desktop/fiap/aichatbot/gs2/meu_modelo_serializado_class.pickle', 'rb') as f:
    modelo = pickle.load(f)

# Rota para receber os dados e fazer previsões doença cardíaca 
@app.route('/prever1', methods=['GET'])
def prever1():
    # Obter parâmetros da solicitação GET										
    parametro1 = float(request.args.get('Age'))
    parametro2 = float(request.args.get('Sex'))
    parametro3 = float(request.args.get('ChestPainType'))
    parametro4 = float(request.args.get('RestingBP'))
    parametro5 = float(request.args.get('Cholesterol'))
    parametro6 = float(request.args.get('FastingBS'))
    parametro7 = float(request.args.get('RestingECG'))
    parametro8 = float(request.args.get('MaxHR'))
    parametro9 = float(request.args.get('ExerciseAngina'))
    parametro10 = float(request.args.get('Oldpeak'))
    parametro11 = float(request.args.get('ST_Slope'))

    # Fazer previsões usando o modelo 
    entrada = np.array([[parametro1, parametro2, parametro3, parametro4, parametro5, 
    parametro6, parametro7, parametro8, parametro9, parametro10, parametro11]])
    resultado = modelo.predict(entrada)

    # Retornar o resultado como JSON
    return jsonify({'previsao se possui doença cardíaca ou não': resultado.tolist()})

# Segundo modelo

# Aqui estamos carregando o modelo já treinado que está no arquivo JobLib
with open('C:/Users/HP/Desktop/fiap/aichatbot/gs2/meu_modelo_serializado_class2.pickle', 'rb') as f:
    modelo = pickle.load(f)

# Rota para receber os dados e fazer previsões exercise angina
@app.route('/prever2', methods=['GET'])
def prever2():
    # Obter parâmetros da solicitação GET
    parametro1 = float(request.args.get('Age'))
    parametro2 = float(request.args.get('Sex'))
    parametro3 = float(request.args.get('ChestPainType'))
    parametro4 = float(request.args.get('RestingBP'))
    parametro5 = float(request.args.get('Cholesterol'))
    parametro6 = float(request.args.get('FastingBS'))
    parametro7 = float(request.args.get('RestingECG'))
    parametro8 = float(request.args.get('MaxHR'))
    parametro9 = float(request.args.get('Oldpeak'))
    parametro10 = float(request.args.get('ST_Slope'))
    parametro11 = float(request.args.get('HeartDisease'))


    # Fazer previsões usando o modelo 
    entrada = np.array([[parametro1, parametro2, parametro3, parametro4, parametro5, 
    parametro6, parametro7, parametro8, parametro9, parametro10, parametro11]])
    resultado = modelo.predict(entrada)

    # Retornar o resultado como JSON
    return jsonify({'previsao do exercise angina': resultado.tolist()})

if __name__ == '__main__':
    print("Servidor Flask em execução")
    # Executar o aplicativo Flask
    app.run(debug=True)
