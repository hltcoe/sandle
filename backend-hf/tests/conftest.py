from hypothesis import settings, Phase

settings.register_profile('explicit', phases=(Phase.explicit,))
