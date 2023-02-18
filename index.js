import express from 'express';
import { config } from 'dotenv';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { Configuration, OpenAIApi } from 'openai';

const app = express();
config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const configuration = new Configuration({
    apiKey: process.env.OPEN_AI_API
});
const openai = new OpenAIApi(configuration);

app.use(express.json());
app.use(express.static(join(__dirname, 'public')));

app.post('/ask', async (req, res) => {
    try {
        const prompt = req.body.prompt;
        const response = await openai.createCompletion({
            model: "text-davinci-003",
            prompt: prompt,
            temperature: 0.9,
            max_tokens: 150,
            top_p: 1,
            frequency_penalty: 0.6,
            presence_penalty: 0.6
        });
        const moreques = await openai.createCompletion({
            model: "text-davinci-003",
            prompt: `Give me some more questions like ${prompt}`,
            temperature: 0.9,
            max_tokens: 150,
            top_p: 1,
            frequency_penalty: 0.6,
            presence_penalty: 0.6
        });

        res.send({ answer: response.data.choices[0].text, moreque: moreques.data.choices[0].text });

    } catch (err) {
        console.log(err);
        res.status(500).send({ message: err });
    }

});

// create a new instance of SpeechSynthesisUtterance
// const utterance = new SpeechSynthesisUtterance();

// // set the text to be spoken
// utterance.text = "Hello, world!";

// // set the voice
// utterance.voice = speechSynthesis.getVoices()[0]; // change the index to select a different voice

// // speak the text
// speechSynthesis.speak(utterance);
  


const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Listening on http://localhost:${port}`));