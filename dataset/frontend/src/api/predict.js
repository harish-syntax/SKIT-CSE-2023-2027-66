import { atmZones } from "../data/zones";

// MOCK predictor so the UI works standalone.
// When Lakshita's endpoint is ready, replace the body with:
//
//   const res = await fetch("http://localhost:8000/predict", {
//     method: "POST",
//     headers: { "Content-Type": "application/json" },
//     body: JSON.stringify(complaint),
//   });
//   return res.json(); // expected: [{ id, name, lat, lng, probability }, x3]

export async function predictTopZones(complaint) {
  await new Promise((r) => setTimeout(r, 700));

  const seed =
    complaint.fraudType.length * 7 + Math.floor(Number(complaint.amount) / 1000);

  return atmZones
    .map((z, i) => ({
      ...z,
      probability: Math.min(0.97, 0.35 + ((seed * (i + 3)) % 60) / 100),
    }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 3);
}
