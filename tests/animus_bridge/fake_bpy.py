"""A minimal fake bpy module for testing the Animus bridge without Blender.

Models just enough of the data API: objects, armatures with bones,
actions with FCurves and custom properties, animation data with NLA
tracks and strips, and a manual timers registry.

Every data access records the calling thread id in ACCESS_THREADS so
tests can prove that bpy is only touched from the thread that pumps the
timer, never from a socket thread.
"""

import threading

ACCESS_THREADS = set()


def _touch():
    ACCESS_THREADS.add(threading.get_ident())


class FakeKeyframePoints:
    def __init__(self):
        self.points = []

    def insert(self, frame, value):
        _touch()
        self.points.append((frame, value))
        return (frame, value)

    def __len__(self):
        return len(self.points)


class FakeFCurve:
    def __init__(self, data_path, index):
        self.data_path = data_path
        self.array_index = index
        self.keyframe_points = FakeKeyframePoints()


class FakeFCurves:
    def __init__(self):
        self._curves = []

    def new(self, data_path, index=0):
        _touch()
        if self.find(data_path, index=index) is not None:
            raise RuntimeError(f"fcurve already exists: {data_path}[{index}]")
        curve = FakeFCurve(data_path, index)
        self._curves.append(curve)
        return curve

    def find(self, data_path, index=0):
        _touch()
        for curve in self._curves:
            if curve.data_path == data_path and curve.array_index == index:
                return curve
        return None

    def __iter__(self):
        return iter(self._curves)

    def __len__(self):
        return len(self._curves)


class FakeAction:
    def __init__(self, name):
        self.name = name
        self.fcurves = FakeFCurves()
        self.use_fake_user = False
        self._props = {}

    def __setitem__(self, key, value):
        _touch()
        self._props[key] = value

    def __getitem__(self, key):
        _touch()
        return self._props[key]

    def get(self, key, default=None):
        _touch()
        return self._props.get(key, default)

    def key_snapshot(self):
        """Test helper. Not part of the bpy surface."""
        return {
            (curve.data_path, curve.array_index): list(curve.keyframe_points.points)
            for curve in self.fcurves
        }


class FakeNamedCollection:
    """Shared behavior for bpy.data.actions and bpy.data.objects."""

    def __init__(self):
        self._items = {}

    def _dedup(self, name):
        if name not in self._items:
            return name
        number = 1
        while f"{name}.{number:03d}" in self._items:
            number += 1
        return f"{name}.{number:03d}"

    def get(self, name):
        _touch()
        return self._items.get(name)

    def keys(self):
        _touch()
        return list(self._items.keys())

    def remove(self, item):
        _touch()
        for key, value in list(self._items.items()):
            if value is item:
                del self._items[key]
                return
        raise RuntimeError("item not in collection")

    def __iter__(self):
        return iter(self._items.values())

    def __len__(self):
        return len(self._items)


class FakeActions(FakeNamedCollection):
    def new(self, name):
        _touch()
        action = FakeAction(self._dedup(name))
        self._items[action.name] = action
        return action


class FakeObjects(FakeNamedCollection):
    def add(self, obj):
        self._items[obj.name] = obj
        return obj


class FakeBone:
    def __init__(self, name):
        self.name = name


class FakeBones:
    def __init__(self, names):
        self._bones = {name: FakeBone(name) for name in names}

    def keys(self):
        _touch()
        return list(self._bones.keys())

    def get(self, name):
        _touch()
        return self._bones.get(name)

    def __iter__(self):
        return iter(self._bones.values())


class FakeArmature:
    def __init__(self, name, bone_names):
        self.name = name
        self.bones = FakeBones(bone_names)


class FakeNlaStrip:
    def __init__(self, name, start, action):
        self.name = name
        self.action = action
        self.frame_start = float(start)
        frames = [frame for keys in action.key_snapshot().values() for frame, _ in keys]
        span = (max(frames) - min(frames)) if frames else 0
        self.frame_end = float(start) + float(span)


class FakeNlaStrips:
    def __init__(self):
        self._strips = []

    def new(self, name, start, action):
        _touch()
        for strip in self._strips:
            if strip.name == name:
                raise RuntimeError(f"strip name already used: {name}")
        strip = FakeNlaStrip(name, start, action)
        self._strips.append(strip)
        return strip

    def __iter__(self):
        return iter(self._strips)

    def __len__(self):
        return len(self._strips)


class FakeNlaTrack:
    _counter = 0

    def __init__(self):
        FakeNlaTrack._counter += 1
        self.name = f"NlaTrack.{FakeNlaTrack._counter:03d}"
        self.strips = FakeNlaStrips()


class FakeNlaTracks:
    def __init__(self):
        self._tracks = []

    def new(self, prev=None):
        _touch()
        track = FakeNlaTrack()
        self._tracks.append(track)
        return track

    def __iter__(self):
        return iter(self._tracks)

    def __len__(self):
        return len(self._tracks)


class FakeAnimData:
    def __init__(self):
        self.action = None
        self.nla_tracks = FakeNlaTracks()


class FakeObject:
    def __init__(self, name, obj_type, data):
        self.name = name
        self.type = obj_type
        self.data = data
        self.animation_data = None

    def animation_data_create(self):
        _touch()
        if self.animation_data is None:
            self.animation_data = FakeAnimData()
        return self.animation_data


class FakeData:
    def __init__(self):
        self.actions = FakeActions()
        self.objects = FakeObjects()


class FakeTimers:
    """Manual timer registry. Tests call pump() to run callbacks."""

    def __init__(self):
        self._functions = []

    def register(self, function, first_interval=0.0, persistent=False):
        self._functions.append(function)

    def is_registered(self, function):
        return function in self._functions

    def unregister(self, function):
        self._functions.remove(function)

    def pump(self):
        """Run every registered callback once, honoring the None contract."""
        for function in list(self._functions):
            result = function()
            if result is None and function in self._functions:
                self._functions.remove(function)


class FakeApp:
    def __init__(self):
        self.background = False
        self.timers = FakeTimers()


data = FakeData()
app = FakeApp()


def reset():
    """Fresh scene data, fresh timers, fresh access log."""
    global data, app
    data = FakeData()
    app = FakeApp()
    ACCESS_THREADS.clear()
    FakeNlaTrack._counter = 0


def add_armature_object(name, bone_names):
    """Test helper: put an armature object into the fake scene."""
    armature = FakeArmature(f"{name}_rig", bone_names)
    return data.objects.add(FakeObject(name, "ARMATURE", armature))


def add_plain_object(name):
    """Test helper: a non-armature object for refusal tests."""
    return data.objects.add(FakeObject(name, "MESH", None))
