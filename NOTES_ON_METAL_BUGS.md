# On the Metal Bug(s) in MacOS and Why I Can't Continue with MLX Projects

On a number of occasions, I've mentioned the Metal bugs in MacOS and the possibility that MLX might trigger them.  

It appears I may be the only one facing this, which is quite baffling, or perhaps we need more "samples" to verify the issue. These bugs seem to manifest in high-end machines equipped with numerous external devices, for some odd reason, like maxing out 4 Thunderbolt ports and so on.

Initially, I gave it the benefit of the doubt, swapping my Mac Studio M2 Ultras between my music studio and my office as a last resort. Sadly, just a day later, the one from my music studio began exhibiting the same Metal bugs, with Final Cut Pro refusing to render, among other issues. The only solutions are reboots or force quitting some of the GPU processes, almost as if something needs to trigger a reset of the Metal engine.

I had hoped the issue with the one in my office was an isolated case, but it turns out that was not the case.

The chances of both Mac Studio M2s being defective seems nearly impossible unless there's an underlying design flaw.

Why I appear to be the lone voice raising this issue remains a mystery.

I want to make it clear: the bug is very much real and present. It may escalate into a more significant issue. Frankly, my patience is wearing thin; working with anything MLX on my Mac Studio M2s has become thoroughly unenjoyable.

Both M2 Ultras share the same specs, with 192GB of RAM. Now, merely running PyCharm with an MLX virtual env triggers this bug. I've not even executed any MLX code. I was merely writing an essay in PyCharm in Markdown, tried rendering some AI-generated images and audio, and then Final Cut Pro balked at rendering after a session of fine edits. Terminating PyCharm (sometimes requiring the shutdown of other GPU processes via Activity Monitor) and restarting it usually resolves the issue. If not, a reboot is required.

I've already reported this issue to Apple through the feedback app, but as you might expect, I'm not holding my breath for any reply given the niche nature of the problem.

Given the rarity of reported cases, I'm not optimistic about a resolution, if any, to this bug, or bugs. My best wishes go out to those who are enjoying a seamless experience with MLX on their Macs. 

I simply can't invest myself in anything that causes me more headaches than joy. Furthermore, I can't recommend using MLX in production environments, especially if you're working with high-end machines. 

For these reasons, with heavy hearts, I've decided to stop working on MLX projects in this repository. The book and other contents I've already written will remain, but I won't be adding any new MLX-related content. And for most of you, that would more than suffice.

Despite everything, I remain deeply appreciative of the effort and commitment demonstrated by the MLX team and the community. My best wishes are with all of you. It seems I'm the only one who needs to take a step back from this journey. 

Sincerely, 
Wankyu Choi