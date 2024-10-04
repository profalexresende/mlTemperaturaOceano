using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace mlTemperaturaOceano
{


    public class DadosOceano
    {
        [LoadColumn(0)] public float Latitude { get; set; }
        [LoadColumn(1)] public float Longitude { get; set; }
        [LoadColumn(2)] public float Profundidade { get; set; }
        [LoadColumn(3)] public float Salinidade { get; set; }
        [LoadColumn(4)] public string EpocaDoAno { get; set; }
        [LoadColumn(5)] public float Temperatura { get; set; } // Rótulo (valor a ser previsto)
    }

    public class PrevisaoTemperaturaOceano
    {
        [ColumnName("Score")]
        public float TemperaturaPrevista { get; set; }
    }

}
