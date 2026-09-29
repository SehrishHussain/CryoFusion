import asyncio
import aiohttp


URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr/.zgroup"
)


async def main():

    print("===== AIOHTTP TEST =====")
    print("URL:", URL)

    try:

        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:

            async with session.get(URL) as response:

                print("Status:", response.status)

                text = await response.text()

                print("Response:")
                print(text[:500])

    except Exception as e:

        print("ERROR TYPE:", type(e).__name__)
        print("ERROR:", e)


if __name__ == "__main__":
    asyncio.run(main())