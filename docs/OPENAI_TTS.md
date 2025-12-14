# OpenAI TTS Integration

Easy-OWW now supports **OpenAI GPT-4o-mini-TTS** for high-quality voice synthesis, providing better voice diversity and naturalness compared to Piper TTS.

## Overview

The integration automatically uses OpenAI TTS when available, with graceful fallback to Piper TTS if:
- `OPENAI_API_KEY` environment variable is not set
- OpenAI API fails or is unavailable
- Dependencies are not installed

## Features

- **10 diverse voices**: alloy, nova, echo, shimmer, onyx, fable, ash, ballad, coral, sage, verse
- **Increased adversarial negatives**: 5000 samples (up from 3000) for better false positive reduction
- **Cost tracking**: Real-time cost estimation and reporting
- **Speed variations**: 0.8x to 1.3x for natural diversity
- **Automatic fallback**: Seamlessly falls back to Piper TTS if unavailable

## Setup

### 1. Install Dependencies

```bash
pip install openai>=1.0.0 pydub>=0.25.0
```

**Note**: `pydub` requires `ffmpeg` to be installed on your system:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### 2. Set API Key

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or add it to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### 3. Verify Setup

The training process will automatically detect and use OpenAI TTS:

```bash
easy-oww train my-wake-word
```

You should see messages like:
```
[cyan]Using OpenAI GPT-4o-mini-TTS for synthetic samples[/cyan]
Estimated cost: $0.0298 for 298,000 characters
```

## Cost Estimation

### Pricing

OpenAI GPT-4o-mini-TTS costs approximately **$0.10 per 1M characters**.

### Typical Training Run Costs

For a standard wake word model:

| Component | Samples | Characters | Cost |
|-----------|---------|------------|------|
| Positive samples (synthetic) | 980 | ~60,000 | $0.006 |
| Adversarial negatives | 5,000 | ~240,000 | $0.024 |
| **Total** | **5,980** | **~300,000** | **$0.030** |

**Total cost per training run: ~$0.03** (well under the $1.00 budget)

Even with 10,000 adversarial negatives, the cost would only be ~$0.06 per training run.

## Usage

### Automatic Mode (Recommended)

Simply train your model as usual. OpenAI TTS will be used automatically if configured:

```bash
easy-oww train "hey assistant"
```

### Manual Control

To disable OpenAI TTS and force Piper TTS usage, unset the API key temporarily:

```bash
unset OPENAI_API_KEY
easy-oww train "hey assistant"
```

## Voice Selection

The system automatically selects **10 diverse voices** for maximum variety:

1. **alloy** - Neutral, balanced
2. **nova** - Female, energetic
3. **echo** - Male, clear
4. **shimmer** - Female, soft
5. **onyx** - Deep male
6. **fable** - British accent, warm
7. **ash** - Male, conversational
8. **coral** - Female, warm
9. **sage** - Male, professional
10. **ballad** - Female, expressive
11. **verse** - Male, narrative

Voices are rotated across samples to maximize speaker diversity.

## Output Example

When training with OpenAI TTS, you'll see detailed cost tracking:

```
Generating adversarial phrases for 'hey assistant'...
  Generated 5000 unique adversarial phrases
  Using 10 OpenAI voices for TTS synthesis
  Estimated cost: $0.0240 for 240,000 characters
  Synthesizing adversarial samples with OpenAI GPT-4o-mini-TTS...
  This will create 5000 audio clips (may take 15-20 minutes)
  [████████████████████████████████████████] 5000/5000
  ✓ Generated 5000 adversarial samples
  TTS Usage: 5,000 requests, 240,234 characters
  Total Cost: $0.0240
```

## Benefits Over Piper TTS

| Feature | OpenAI TTS | Piper TTS |
|---------|------------|-----------|
| Voice Count | 10 voices | 2-5 voices (if downloaded) |
| Voice Quality | Very high (human-like) | Good (synthetic) |
| Naturalness | Excellent | Good |
| Setup Complexity | API key only | Voice model downloads |
| Cost | ~$0.03/training | Free |
| Speed | Moderate (API calls) | Fast (local) |
| Offline Support | No | Yes |

## Troubleshooting

### Error: "OPENAI_API_KEY not set"

Set your API key:
```bash
export OPENAI_API_KEY='your-key'
```

### Error: "pydub package required"

Install pydub and ffmpeg:
```bash
pip install pydub
brew install ffmpeg  # macOS
```

### Error: "OpenAI API rate limit exceeded"

The system generates samples sequentially to respect rate limits. If you hit limits:
- Wait a few minutes and retry
- Consider using Piper TTS as fallback
- Contact OpenAI to increase your rate limits

### Cost Concerns

If estimated cost exceeds your budget:
1. The system will warn you before generation
2. You can manually reduce adversarial sample count in code
3. Use Piper TTS instead (free, offline)

## Configuration

### Changing Voice Count

Edit `/easy_oww/training/full_trainer.py`:

```python
# Use fewer voices (e.g., 5)
voices = OpenAITTS.get_diverse_voices(count=5)

# Or use all 11 voices
voices = OpenAITTS.get_diverse_voices(count=11)
```

### Changing Adversarial Count

Edit `/easy_oww/training/full_trainer.py`:

```python
# Generate fewer adversarial samples (e.g., 3000)
adversarial_texts = self._generate_simple_adversarial_texts(self.wake_word, n=3000)

# Or more (e.g., 10000)
adversarial_texts = self._generate_simple_adversarial_texts(self.wake_word, n=10000)
```

## API Reference

### OpenAITTS Class

```python
from easy_oww.tts import OpenAITTS

# Initialize
tts = OpenAITTS(api_key='optional-key')

# Generate speech
tts.generate_speech(
    text="hello world",
    voice="alloy",
    output_path=Path("output.wav"),
    sample_rate=16000,
    speed=1.0
)

# Get usage stats
stats = tts.get_usage_stats()
# {'total_requests': 100, 'total_characters': 5000,
#  'estimated_cost_usd': 0.0005, 'cost_per_request': 0.000005}

# Estimate cost before generation
cost = tts.estimate_cost(character_count=100000)
# Returns: 0.01
```

## Best Practices

1. **Set API key globally**: Add to shell profile for convenience
2. **Monitor costs**: Check usage stats after training runs
3. **Use for production**: OpenAI TTS significantly improves model quality
4. **Keep Piper as backup**: Maintain Piper installation for offline work
5. **Test both**: Compare model performance with both TTS systems

## FAQ

**Q: Is OpenAI TTS required?**
A: No, it's optional. The system falls back to Piper TTS automatically.

**Q: Can I mix OpenAI and Piper voices?**
A: Not currently. The system uses one or the other per training run.

**Q: How much does it cost for multiple training runs?**
A: Each training run costs ~$0.03, so 10 runs = ~$0.30, 100 runs = ~$3.00.

**Q: Will costs increase in the future?**
A: OpenAI may adjust pricing. Check current rates at https://openai.com/pricing

**Q: Can I use GPT-4 TTS instead of GPT-4o-mini?**
A: Currently only GPT-4o-mini-TTS is supported for cost efficiency. You can modify the code to use other models.

## Changelog

### v0.2.0 (Current)
- Added OpenAI GPT-4o-mini-TTS integration
- Increased adversarial negatives from 3000 to 5000
- Added 10 diverse voice support
- Added real-time cost tracking and reporting
- Automatic fallback to Piper TTS

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/easy-oww/issues
- Check logs for detailed error messages
- Verify API key and dependencies are correctly set up
