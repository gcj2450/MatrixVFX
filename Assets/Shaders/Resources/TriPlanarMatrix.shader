Shader "Unlit/TriPlanarMatrix"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
         Tags { "Queue" = "Transparent" "IgnoreProjector" = "True" "RenderType" = "Transparent"}
         LOD 100
         Blend One One              //1st change here
         Cull Off
         ZWrite Off
         
         Pass
         {
             CGPROGRAM
             #pragma vertex   vert
             #pragma fragment frag
             #include "UnityCG.cginc"
             #define vec2 float2
            #define vec3 float3
            #define vec4 float4
            #define time _Time.g
            #define fract frac

             struct appdata
             {
                 float4 vertex : POSITION;
                 float2 uv     : TEXCOORD0;
                 float3 normal : NORMAL;
             };
             
             struct v2f
             {
                 //float2 uv       : TEXCOORD0;
                 float4 vertex    : SV_POSITION;
                 float3 worldPos  : TEXCOORD0;
                 float3 normal    : NORMAL;
                 float4 screenPos : TEXCOORD2;
             };
             
             sampler2D _MainTex;
             float4    _MainTex_ST;
             // -----------------------------------
             v2f vert (appdata v)
             {
                 v2f o;
                 o.vertex    = UnityObjectToClipPos(v.vertex);
                 //o.uv      = TRANSFORM_TEX(v.uv, _MainTex);
                 o.worldPos  = mul(unity_ObjectToWorld, v.vertex);
                 o.normal    = UnityObjectToWorldNormal(v.normal);
                 o.screenPos = ComputeScreenPos(o.vertex);
                 return o;
             }
             
             sampler2D global_white_noise;
             sampler2D global_font_texture;
             uint      global_colored;
              //---------------------------------------------------------

              //==========================
              float length2(vec2 p) { return dot(p, p); }

                float noise(vec2 p){
                    return fract(sin(fract(sin(p.x) * (45.0)) + p.y) * 30.0);
                }

                float worley(vec2 p) {
                    float d = 1e30;
                    for (int xo = -1; xo <= 1; ++xo) {
                        for (int yo = -1; yo <= 1; ++yo) {
                            vec2 tp = floor(p) + vec2(xo, yo);
                            d = min(d, length2(p - tp - vec2(noise(tp),noise(tp))));
                        }
                    }
                    return 3.0*exp(-3.0*abs(2.0*d - 1.0));
                }

                float fworley(vec2 p) {
                    return sqrt(sqrt(sqrt(
                        1.1 * // light
                        worley(p*5. + .3 + time*.0525) *
                        sqrt(worley(p * 50. + 0.3 + time * -0.15)) *
                        sqrt(sqrt(worley(p * -10. + 9.3))))));
                }

                float3 mainimage(float2 fragCoord)
                {
                    vec2 uv =normalize( fragCoord);
                    float t = fworley(uv);
                    t *= exp(-length2(abs(0.7*uv - 1.0)));
                   return float3(t * vec3(0.1, 1.5*t, 1.2*t + pow(t, 0.5-t)));
                    
                }

                float random (half2 st) {
                    return frac(sin(dot(st.xy,half2(12.9898,78.233)))*43758.5453123);
                }

                vec4 mainImage2(in vec2 fragCoord )
                {
                    vec2 iResolution=vec2(100.0,100.0);
                    vec2 spiralCenter = iResolution.xy / 2.0;
                    float abstandSpiralCenter = distance(fragCoord, spiralCenter);
                    float abstandSpiralCenterNorm = abstandSpiralCenter / length(iResolution.xy / 2.0);
    
                    float winkel = sqrt(abstandSpiralCenterNorm) * 10.0 * sin(_Time * .17)   + _Time * .61;
                    vec2 vergleichspunkt = spiralCenter + abstandSpiralCenter * vec2(sin(winkel), cos(winkel));
                    float abstandVergleichspunkt = distance(fragCoord, vergleichspunkt);
                    float abstandVergleichspunktNorm = abstandVergleichspunkt / length(iResolution.xy / 2.0);
                    float subtrahend = abstandVergleichspunktNorm / abstandSpiralCenterNorm;

                    float winkel2 = sqrt(abstandSpiralCenterNorm) * 10.0 * sin(_Time * .23 + .1)   + _Time * .31;
                    vec2 vergleichspunkt2 = spiralCenter + abstandSpiralCenter * vec2(sin(winkel2), cos(winkel2));
                    float abstandVergleichspunkt2 = distance(fragCoord, vergleichspunkt2);
                    float abstandVergleichspunktNorm2 = abstandVergleichspunkt2 / length(iResolution.xy / 2.0);
                    float subtrahend2 = abstandVergleichspunktNorm2 / abstandSpiralCenterNorm;

                    float winkel3 = sqrt(abstandSpiralCenterNorm) * 10.0 * sin(_Time * .41 + .62)   + _Time * .47;
                    vec2 vergleichspunkt3 = spiralCenter + abstandSpiralCenter * vec2(sin(winkel3), cos(winkel3));
                    float abstandVergleichspunkt3 = distance(fragCoord, vergleichspunkt3);
                    float abstandVergleichspunktNorm3 = abstandVergleichspunkt3 / length(iResolution.xy / 2.0);
                    float subtrahend3 = abstandVergleichspunktNorm3 / abstandSpiralCenterNorm;

                    float winkel4 = sqrt(abstandSpiralCenterNorm) * 10.0 * sin(_Time * .38 + .17)   + _Time * .85;
                    vec2 vergleichspunkt4 = spiralCenter + abstandSpiralCenter * vec2(sin(winkel4), cos(winkel4));
                    float abstandVergleichspunkt4 = distance(fragCoord, vergleichspunkt4);
                    float abstandVergleichspunktNorm4 = abstandVergleichspunkt4 / length(iResolution.xy / 2.0);
                    float subtrahend4 = abstandVergleichspunktNorm4 / abstandSpiralCenterNorm;

                    float winkel5 = sqrt(abstandSpiralCenterNorm) * 10.0 * sin(_Time * .48 + .95)   + _Time * .57;
                    vec2 vergleichspunkt5 = spiralCenter + abstandSpiralCenter * vec2(sin(winkel5), cos(winkel5));
                    float abstandVergleichspunkt5 = distance(fragCoord, vergleichspunkt5);
                    float abstandVergleichspunktNorm5 = abstandVergleichspunkt5 / length(iResolution.xy / 2.0);
                    float subtrahend5 = abstandVergleichspunktNorm5 / abstandSpiralCenterNorm;

                    float winkel6 = sqrt(abstandSpiralCenterNorm) * 10.0 * sin(_Time * .29 + .27)   + _Time * .54;
                    vec2 vergleichspunkt6 = spiralCenter + abstandSpiralCenter * vec2(sin(winkel6), cos(winkel6));
                    float abstandVergleichspunkt6 = distance(fragCoord, vergleichspunkt6);
                    float abstandVergleichspunktNorm6 = abstandVergleichspunkt6 / length(iResolution.xy / 2.0);
                    float subtrahend6 = abstandVergleichspunktNorm6 / abstandSpiralCenterNorm;

                    vec3 fragColor1 = vec3(2.0 - abstandVergleichspunktNorm - abstandVergleichspunktNorm4 - abstandVergleichspunktNorm6, 2.0 - abstandVergleichspunktNorm2 - abstandVergleichspunktNorm5 - abstandVergleichspunktNorm4, 2.0 - abstandVergleichspunktNorm3 - abstandVergleichspunktNorm6 - abstandVergleichspunktNorm5);
                    vec3 fragColor2 = vec3(4.0 - subtrahend - subtrahend4 - subtrahend6, 4.0 - subtrahend2 - subtrahend5 - subtrahend4, 4.0 - subtrahend3 - subtrahend6 - subtrahend5);
                    float faktor = tex2D(global_font_texture,vec2(0,0)).x;
                    faktor = pow(faktor, 5.0);
    
                    // Output to screen
                    return vec4(lerp(fragColor1, fragColor2, faktor), 1.0);
                    //fragColor = vec4(fragColor1, 1.0);
                }

              //==========================

             
             float text(float2 coord)
             {
                 float2 uv    = frac (coord.xy/ 16.);                // Geting the fract part of the block, this is the uv map for the blocl
                 float2 block = floor(coord.xy/ 16.);                // Getting the id for the block. The first blocl is (0,0) to its right (1,0), and above it (0,1) 
                        uv    = uv * 0.7 + .1;                       // Zooming a bit in each block to have larger ltters
                    
                    float2 rand  = tex2D(global_white_noise,         // This texture contains animated white noise. The white noise is animated in compute shaders
                                   block.xy/float2(512.,512.)).xy;   // 512 is the white noise texture width. This division ensures that each block samples exactly one pixel of the noise texture
                 
                        rand  = floor(rand*16.);                     // Each random value is used for the block to sample one of the 16 columns of the font texture. This rand offset is what picks the letter, the animated white noise is what changes it
                        uv   += rand;                                // The random texture has a different value und the xy channels. This ensures that randomly one member of the texture is picked 
                 
                        uv   *= 0.0625;                              // So far the uv value is between 0-16. To sample the font texture we need to normalize this to 0-1. hence a divid by 16
                        uv.x  = -uv.x;
                   return tex2D(global_font_texture, uv).r;
             }
             
             //---------------------------------------------------------
#define dropLength 512
         float3 rain(float2 fragCoord)
         {
               fragCoord.x  = floor(fragCoord.x/ 16.);             // This is the exact replica of the calculation in text function for getting the cell ids. Here we want the id for the columns 
               
               float offset = sin (fragCoord.x*15.);               // Each drop of rain needs to start at a different point. The column id  plus a sin is used to generate a different offset for each columm
               float speed  = cos (fragCoord.x*3.)*.15 + .35;      // Same as above, but for speed. Since we dont want the columns travelling up, we are adding the 0.7. Since the cos *0.3 goes between -0.3 and 0.3 the 0.7 ensures that the speed goes between 0.4 mad 1.0. This is also control parameters for min and max speed
               float y      = frac((fragCoord.y / dropLength)      // This maps the screen again so that top is 1 and button is 0. The addition with time and frac would cause an entire bar moving from button to top
                                + _Time.y * speed + offset);       // the speed and offset would cause the columns to move down at different speeds. Which causes the rain drop effect
               
               // return float3(.1, 1., .35) / (y*20.);               // adjusting the retun color based on the columns calculations. 
               //这个彩色好看点
                return float3(random(float2(offset,offset)),random(float2(offset,offset)*y),random(float2(offset,offset)*speed)) / (y*5.);
         }

           //---------------------------------------------------------

           uint _session_rand_seed; // required by the RandomLi Include
#include "RandomLib.cginc"
#include "LabColorspace.cginc"
         float3 rain_colored(float2 fragCoord)
         {
             fragCoord.x  = floor(fragCoord.x/ 16.);               // This is the exact replica of the calculation in text function for getting the cell ids. Here we want the id for the columns 
             
             float offset = rnd (fragCoord.x*521., 612);           // Each drop of rain needs to start at a different point. The column id  plus a sin is used to generate a different offset for each columm
             float speed  = rnd (fragCoord.x*612., 951)*.15 + .35; // Same as above, but for speed. Since we dont want the columns travelling up, we are adding the 0.7. Since the cos *0.3 goes between -0.3 and 0.3 the 0.7 ensures that the speed goes between 0.4 mad 1.0. This is also control parameters for min and max speed
                   speed *= 0.4;
             float y      = frac((fragCoord.y / dropLength)        // This maps the screen again so that top is 1 and button is 0. The addition with time and frac would cause an entire bar moving from button to top
                                 + _Time.y * speed + offset);      // the speed and offset would cause the columns to move down at different speeds. Which causes the rain drop effect
             
             
             int    randomSeed = (fragCoord.x +
                                 floor((fragCoord.y / dropLength) + _Time.y * speed + offset)) *51.;
                
             float3 col = float3(rnd(randomSeed, 21),
                                 frac(rnd(randomSeed, 712)+0.8), 
                                 frac(rnd(randomSeed, 61)+0.2));
                    col = lab2rgb(col);
             return col / (y*20.);                                 // adjusting the retun color based on the columns calculations. 
         }
         
         //---------------------------------------------------------
#define scale     0.2
#define sharpness 10.
         float3 MatrixEffect(float2 coord) {
             float3      col = float3(0., 0., 0.);
             float3 rain_col = rain(coord* float2(dropLength, dropLength)*scale);
             
             // if (global_colored == 1)
             //     rain_col = rain_colored(coord * float2(dropLength, dropLength)*scale);
             return                 text(coord * float2(dropLength, dropLength)*scale) *rain_col;
             //这里可以替换成任意颜色
             // return                 mainImage2(coord * float2(dropLength, dropLength)*scale) ;
         }
         
         float     _Global_Transition_value;
         float3    _Global_Effect_center;

#include "Transition.cginc"
         // -----------------------------------
         fixed4 frag (v2f i) : SV_Target
         {
             fixed4 col      = float4(0.,0.,0.,1.);
             float3 colFront = MatrixEffect(i.worldPos.xy + sin(i.worldPos.zz));
             float3 colSide  = MatrixEffect(i.worldPos.zy + sin(i.worldPos.xx));
             float3 colTop   = MatrixEffect(i.worldPos.xz + sin(i.worldPos.yy));
             
             float3 blendWeight  = pow(normalize(abs(i.normal)), sharpness);
                    blendWeight /= (blendWeight.x+ blendWeight.y+ blendWeight.z);
                    col.xyz      = colFront * blendWeight.z + 
                                   colSide  * blendWeight.x + 
                                   colTop   * blendWeight.y;
             
             float distance_to_center = distance(i.worldPos.xyz, _Global_Effect_center.xyz);
             float control_value      = saturate(_Global_Transition_value);
             if (control_value * 60.0f < distance_to_center) col = col * 0.0f;
             
             float2 screenPos = i.screenPos.xy / i.screenPos.w;
                    col      *= split_from_midle(screenPos.x, _Global_Transition_value, 0.0f);
                    col       = min(1.5,col);
             return col;
         }
         ENDCG
         }
    }
}
