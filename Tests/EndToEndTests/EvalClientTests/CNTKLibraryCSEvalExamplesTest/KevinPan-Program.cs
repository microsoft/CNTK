using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using CNTK;

namespace NewsInsightEvaluation
{
    class KevinProgram
    {
        public static void KevinMain()
        {

#if ORIG
            Data.Init(@"\\stchost-50\ml\cntk\sample\NewsInsight\trained_model\src.l3g.txt");

            var self_attention_version = new ModelEvaluation
            {
                model_file = @"\\stchost-50\ml\cntk\sample\NewsInsight\trained_model\cntk_2_0_6_layer_self_attention_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_01_model_batch_600000_38951002.dnn",
                name = "cntk_2_0_6_layer_self_attention_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_01_model_batch_600000_38951002"
            };
            self_attention_version.Init();
            Console.WriteLine("complete loading model: {0}", self_attention_version.name);

            var without_attention = new ModelEvaluation
            {
                model_file = @"\\stchost-50\ml\cntk\sample\NewsInsight\trained_model\cntk_2_0_6_layer_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_50_model_batch_900000_58410932.dnn",
                name = "cntk_2_0_6_layer_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_50_model_batch_900000_58410932"
            };
            without_attention.Init();
            Console.WriteLine("complete loading model: {0}", without_attention.name);

            Validation v = new Validation
            {
                raw_data_path = @"\\stchost-50\ml\cntk\sample\NewsInsight\trained_model\validate",
                validation_sample_count = 50,
                model_parallel_count = 20,
                negative_document_count = 1
            };
            v.BuildSamples();

            v.Validate(self_attention_version);

            v.Validate(without_attention);
#endif

            Data.Init(@"E:\CNTKMisc\KevinPan-Memory\trained_model\src.l3g.txt");

            var self_attention_version = new ModelEvaluation
            {
                model_file = @"E:\CNTKMisc\KevinPan-Memory\trained_model\cntk_2_0_6_layer_self_attention_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_01_model_batch_600000_38951002.dnn",
                name = "cntk_2_0_6_layer_self_attention_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_01_model_batch_600000_38951002"
            };
            self_attention_version.Init();
            Console.WriteLine("complete loading model: {0}", self_attention_version.name);

            

            Validation v = new Validation
            {
                raw_data_path = @"E:\CNTKMisc\KevinPan-Memory\trained_model\validate",
                validation_sample_count = 50,
                model_parallel_count = 5, // 20,
                negative_document_count = 1
            };
            // v.BuildSamples();

            v.Validate(self_attention_version);

            var without_attention = new ModelEvaluation
            {
                model_file = @"E:\CNTKMisc\KevinPan-Memory\trained_model\cntk_2_0_6_layer_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_50_model_batch_900000_58410932.dnn",
                name = "cntk_2_0_6_layer_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_50_model_batch_900000_58410932"
            };
            without_attention.Init();
            Console.WriteLine("complete loading model: {0}", without_attention.name);

            v.Validate(without_attention);

            //{
            //    //var pq = "2017 cholistan desert rally winners";
            //    //var pd = "12th cholistan desert rally kicks off amidst extravagant fanfare";
            //    var cd = self_attention_version.CosineDistance("google", "microsoft");
            //    Console.WriteLine(cd);

            //    var nq = "airbus biggest plane";
            //    var nd = "call of duty infinite warfare zombies update all ps4 and xbox";

            //    Console.WriteLine("negative = {0}", self_attention_version.CosineDistance(nq, nd));

            //}
        }
    }
}
