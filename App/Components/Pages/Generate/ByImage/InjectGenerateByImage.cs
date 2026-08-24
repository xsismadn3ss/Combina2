using System;
using System.Collections.Generic;
using System.Text;

namespace Combina2.Components.Pages.Generate.ByImage
{
    internal static class InjectGenerateByImage
    {
        /// <summary>
        /// Inyectar servicios para la vista Generate By Image
        /// </summary>
        /// <param name="services"></param>
        /// <returns></returns>
        public static IServiceCollection AddGBI_Services(this IServiceCollection services)
        {
            // Inyectar servicio state machine
            services.AddSingleton<GBI_StateMachine>();

            return services;
        }
    }
}
