using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Http;
using System.Web.Routing;

namespace EvalDll_AzureTutorial
{
    public class WebApiApplication : System.Web.HttpApplication
    {
        protected void Application_Start()
        {
            GlobalConfiguration.Configure(WebApiConfig.Register);

            string pathValue = Environment.GetEnvironmentVariable("PATH");
            string domainBaseDir = AppDomain.CurrentDomain.BaseDirectory;
            string cntkPath = domainBaseDir + @"bin\";
            pathValue += ";" + cntkPath;
            Environment.SetEnvironmentVariable("PATH", pathValue);

        }
    }
}
