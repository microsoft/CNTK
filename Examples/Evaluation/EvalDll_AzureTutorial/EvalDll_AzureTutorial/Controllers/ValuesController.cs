using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Web.Http;
using Swashbuckle.Swagger.Annotations;

using Microsoft.MSR.CNTK.Extensibility.Managed;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;


namespace EvalDll_AzureTutorial.Controllers
{
    public class ValuesController : ApiController
    {
        // GET api/values
        [SwaggerOperation("GetAll")]
        public async Task<IEnumerable<string>> Get()
        {
            return await this.EvaluateCustomDNN("http://3.bp.blogspot.com/-Mwr4UZALiA0/TWBt-3vFR8I/AAAAAAAAA4Y/0tXjI-NhVPM/s1600/j0262568.jpg");
        }

        // GET api/values/5
        [SwaggerOperation("GetById")]
        [SwaggerResponse(HttpStatusCode.OK)]
        [SwaggerResponse(HttpStatusCode.NotFound)]
        public string Get(int id)
        {
            return "value";
        }

        // POST api/values
        [SwaggerOperation("Create")]
        [SwaggerResponse(HttpStatusCode.Created)]
        public void Post([FromBody]string value)
        {
        }

        // PUT api/values/5
        [SwaggerOperation("Update")]
        [SwaggerResponse(HttpStatusCode.OK)]
        [SwaggerResponse(HttpStatusCode.NotFound)]
        public void Put(int id, [FromBody]string value)
        {
        }

        // DELETE api/values/5
        [SwaggerOperation("Delete")]
        [SwaggerResponse(HttpStatusCode.OK)]
        [SwaggerResponse(HttpStatusCode.NotFound)]
        public void Delete(int id)
        {
        }



        public async Task<string[]> EvaluateCustomDNN(string imageUrl)
        {
            string domainBaseDirectory = AppDomain.CurrentDomain.BaseDirectory;
            string workingDirectory = Environment.CurrentDirectory;

            try
            {
                // This example requires the RestNet_18 model.
                // The model can be downloaded from <see cref="https://www.cntk.ai/resnet/ResNet_18.model"/>
                int numThreads = 1;
                List<float> outputs;

                using (var model = new IEvaluateModelManagedF())
                {
                    // initialize the evaluation engine
                    // TODO: initializing the evaluation engine should be one only once at the beginning
                    var initParams = string.Format("numCPUThreads={0}", numThreads);
                    model.Init(initParams);

                    // load the model
                    string modelFilePath = Path.Combine(domainBaseDirectory, @"CNTK\Models\ResNet_18.model");
                    var modelOption = string.Format("modelPath=\"{0}\"", modelFilePath);
                    model.CreateNetwork(modelOption, deviceId: -1);

                    // Prepare input value in the appropriate structure and size
                    var inDims = model.GetNodeDimensions(NodeGroup.Input);
                    if (inDims.First().Value != 224 * 224 * 3)
                    {
                        throw new CNTKRuntimeException(string.Format("The input dimension for {0} is {1} which is not the expected size of {2}.", inDims.First(), inDims.First().Value, 224 * 224 * 3), string.Empty);
                    }

                    // Transform the image
                    System.Net.Http.HttpClient httpClient = new HttpClient();
                    Stream imageStream = await httpClient.GetStreamAsync(imageUrl);
                    Bitmap bmp = new Bitmap(Bitmap.FromStream(imageStream));

                    var resized = bmp.Resize(224, 224, true);
                    var resizedCHW = resized.ParallelExtractCHW();
                    var inputs = new Dictionary<string, List<float>>() { { inDims.First().Key, resizedCHW } };

                    // We can call the evaluate method and get back the results (single layer output)...
                    var outDims = model.GetNodeDimensions(NodeGroup.Output);
                    outputs = model.Evaluate(inputs, outDims.First().Key);
                }

                List<string> o = new List<string>();

                foreach (float f in outputs)
                {
                    o.Add(f.ToString());
                }

                return o.ToArray<string>();
            }
            catch (Exception ex)
            {
                return new string[] { string.Format("domainBase directory {0}, workingDirectory {1}, exception details: {2}.", domainBaseDirectory, workingDirectory, ex.ToString()) };
            }
        }
    }
}
