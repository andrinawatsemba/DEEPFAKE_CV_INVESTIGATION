# UCU Mukono campus set: collection protocol

**Why:** public benchmarks come from elsewhere. This set tests whether a detector trained on them works on phone photos
made around us, including after WhatsApp compression. It is a **pilot from one area** (UCU Mukono and nearby), and we say so.
It is **locked**: no training, no tuning. It is opened once, for the final test.

## Quotas
| What | Per member | Group | Minimum acceptable |
|---|---|---|---|
| Real phone photos | 25 | 150 | 100 |
| AI-generated images | 25 | 150 | 100 |
| Versions of each image | 3 | | original, WhatsApp photo, WhatsApp document |

## Real photos
* **Where:** lecture blocks, library, chapel, cafeteria, hostels, sports field, campus roads and gates, Mukono streets, nearby market stalls.
* **What:** buildings, signs, noticeboards, trees, vehicles including boda bodas, food and stalls, classrooms, red soil, rain and puddles, evening light.
* **Variety on purpose:** different phones, daylight and shade, near and far, indoors and outdoors.
* **Normal camera only.** Switch off beauty mode, portrait mode, HDR enhancers, AI scene optimisers and filters. Do not edit.
  **iPhone:** Settings > Camera > Formats > *Most Compatible* (so photos are JPEG, not HEIC).
* **No identifiable people.** Shoot from a distance or without faces. If a person must appear, get consent first and tick `contains_person` and `consent_obtained`.

## AI images (the matched half)
* For each real scene category, write prompts for the same kind of scene ("a red-roofed lecture block on a tropical university campus, afternoon light, photo").
  This keeps real and AI comparable, so the detector cannot cheat on subject matter.
* Use **at least two different generators across the group**. Record the tool, version or date, and the exact prompt for every image.
* No real people's likeness, no misleading news-style content. Scenes and objects only.
* **Save the original output exactly as downloaded.** Do not crop, resize or convert it.
  (AI tools often give square PNG; phones give large JPEG. That gap is the shortcut we test for, so keep it visible in the log.)

## The three versions (for every image)
1. **original**: the file as it came from the phone or the AI tool > `campus/originals/`
2. **whatsapp_photo**: send the original to a second phone (or "Message yourself") **as a photo**, download it from the chat > `campus/whatsapp_photo/`
3. **whatsapp_document**: send the original **as a document**, download it > `campus/whatsapp_document/`

Same file name in all three folders. Use the same sender and receiver phones for the whole batch.

## Naming and logging
* **ID:** `CAMP_<CODE>_0001` (for example `CAMP_SM_0007`). The name never says real or AI.
* **Labels file** (`metadata/campus_labels_<CODE>.csv`, made by notebook 04): `image_id, label, device_or_tool, prompt_or_scene, contains_person, consent_obtained, notes`.
* Notebook 04 builds `campus_log.csv`, checks all three versions exist, the label is valid, AI images have a prompt, and a person means consent.
* **Consent records** (name, date, "I agree to the photo being used in a university project") are kept privately. They never go on GitHub.
* **Privacy:** campus photos stay in the private shared Drive. They are not committed, not posted, and not shown in the report if a person is identifiable.

## Locking
When every problem is fixed and quotas are met, SM or GW runs the lock cell in notebook 04. It records a SHA-256 for every file in
`campus_manifest.csv`. After that, nobody changes the folder.
