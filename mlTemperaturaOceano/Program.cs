using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using mlTemperaturaOceano;

namespace PrevisaoOceano
{
    class Program
    {
        // Definir caminhos para os arquivos de dados
        static readonly string caminhoDadosTreinamento = Path.Combine(Environment.CurrentDirectory, "Data", "oceano-train.csv");
        static readonly string caminhoDadosTeste = Path.Combine(Environment.CurrentDirectory, "Data", "oceano-test.csv");

        static void Main(string[] args)
        {
            // Passo 1: Criar um novo contexto do ML.NET
            MLContext mlContext = new MLContext();

            // Passo 2: Treinar o modelo
            var modelo = Treinar(mlContext, caminhoDadosTreinamento);

            // Passo 3: Avaliar o modelo
            Avaliar(mlContext, modelo);

            // Passo 4: Testar uma previsão com um exemplo
            TestarPrevisao(mlContext, modelo);
        }

        // Método para treinar o modelo
        public static ITransformer Treinar(MLContext mlContext, string caminhoDados)
        {
            // Carregar os dados de treinamento
            IDataView dadosTreinamento = mlContext.Data.LoadFromTextFile<DadosOceano>(caminhoDados, hasHeader: true, separatorChar: ',');

            // Definir a pipeline do modelo
            var pipeline = mlContext.Transforms.CopyColumns("Label", "Temperatura")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("EpocaDoAnoEncoded", "EpocaDoAno"))
                .Append(mlContext.Transforms.Concatenate("Features", "Latitude", "Longitude", "Profundidade", "Salinidade", "EpocaDoAnoEncoded"))
                .Append(mlContext.Regression.Trainers.FastTree());

            // Treinar o modelo com os dados de treinamento
            var modelo = pipeline.Fit(dadosTreinamento);
            return modelo;
        }

        // Método para avaliar a precisão do modelo
        private static void Avaliar(MLContext mlContext, ITransformer modelo)
        {
            // Carregar os dados de teste
            IDataView dadosTeste = mlContext.Data.LoadFromTextFile<DadosOceano>(caminhoDadosTeste, hasHeader: true, separatorChar: ',');

            // Gerar previsões com o modelo treinado
            var previsoes = modelo.Transform(dadosTeste);

            // Avaliar a precisão do modelo
            var metrics = mlContext.Regression.Evaluate(previsoes, "Label", "Score");

            // Exibir métricas de avaliação
            Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
            Console.WriteLine($"Erro Médio: {metrics.MeanAbsoluteError:#.##}");
        }

        // Método para testar uma previsão com um exemplo
        private static void TestarPrevisao(MLContext mlContext, ITransformer modelo)
        {
            // Criar a função de previsão com o modelo treinado
            var previsaoFunc = mlContext.Model.CreatePredictionEngine<DadosOceano, PrevisaoTemperaturaOceano>(modelo);

            // Exemplo de entrada para prever a temperatura
            var amostra = new DadosOceano()
            {
                Latitude = -23.5f,
                Longitude = 44.2f,
                Profundidade = 5.0f,
                Salinidade = 34.5f,
                EpocaDoAno = "Outono"
            };

            // Fazer a previsão
            var previsao = previsaoFunc.Predict(amostra);
            Console.WriteLine($"Temperatura Prevista: {previsao.TemperaturaPrevista:0.##}");
        }
    }
}
