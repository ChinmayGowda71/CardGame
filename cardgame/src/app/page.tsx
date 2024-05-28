import Image from "next/image";
import table from "../../public/images/table.jpg";
import ClubQ from "../../public/images/clubs_queen.svg";
import ClubK from "../../public/images/clubs_king.svg";
import ClubA from "../../public/images/clubs_ace.svg";
import Club10 from "../../public/images/clubs_10.svg";

export default function Home() {
  return (
    <main className="flex flex-col items-center justify-between">
      <div className="relate w-full">
        <div className="absolute -z-10 w-full">
          <Image
            src={table}
            alt="background image"
            className="w-full"
            width={1000}
            height={1000}
          ></Image>
        </div>
        <div className="absolute left-0 top-1/2 transform -translate-y-1/2 z-10">
          <div className="mx-2">
            <Image //Bottom one
              src={ClubQ}
              alt="Queen of Clubs"
              width={100}
              height={150}
            />
          </div>
        </div>
        <div className="absolute right-0 top-1/2 transform -translate-y-1/2 z-10">
          <div className="mx+2">
            <Image //Top one
              src={ClubK}
              alt="King of Clubs"
              width={100} // Adjust width as necessary
              height={150} // Adjust height as necessary
            />
          </div>
        </div>
      </div>
    </main>
  );
}
