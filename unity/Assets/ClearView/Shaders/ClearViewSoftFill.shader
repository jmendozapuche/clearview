// Soft patch drawn over an ad. Its colour is a blend of the four colours sampled just outside
// the ad's edges, so the patch looks like the surrounding wall instead of a black box.
// Works with the Built-in pipeline and URP (rendered as SRPDefaultUnlit). Single-pass instanced stereo ready.
Shader "ClearView/SoftFill"
{
    Properties
    {
        _ColorL ("Left edge colour", Color) = (0.5, 0.5, 0.5, 1)
        _ColorR ("Right edge colour", Color) = (0.5, 0.5, 0.5, 1)
        _ColorT ("Top edge colour", Color) = (0.5, 0.5, 0.5, 1)
        _ColorB ("Bottom edge colour", Color) = (0.5, 0.5, 0.5, 1)
        _Opacity ("Opacity", Range(0, 1)) = 0
        _Feather ("Feather (fraction of size)", Range(0.01, 0.5)) = 0.12
        _DebugOutline ("Debug outline", Float) = 0
    }

    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType" = "Transparent" "IgnoreProjector" = "True" }

        Pass
        {
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            ZTest LEqual
            Cull Off

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            float4 _ColorL, _ColorR, _ColorT, _ColorB;
            float _Opacity, _Feather, _DebugOutline;

            v2f vert(appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                float u = i.uv.x;
                float v = i.uv.y; // 0 = bottom, 1 = top
                float dh = min(u, 1 - u);
                float dv = min(v, 1 - v);

                // Near the left/right edges use the horizontal blend, near top/bottom the vertical one.
                float wH = dv / (dh + dv + 1e-5);
                float3 horizontal = lerp(_ColorL.rgb, _ColorR.rgb, u);
                float3 vertical = lerp(_ColorB.rgb, _ColorT.rgb, v);
                float3 col = lerp(vertical, horizontal, wH);

                float edge = min(dh, dv);
                float alpha = _Opacity * smoothstep(0, _Feather, edge);

                if (_DebugOutline > 0.5 && edge < 0.02)
                {
                    col = float3(1, 0, 1);
                    alpha = max(alpha, _Opacity);
                }
                return float4(col, alpha);
            }
            ENDCG
        }
    }
}
