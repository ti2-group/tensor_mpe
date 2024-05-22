from collections.abc import Iterable


class Bitset:
    """A custom implementation of a bitset, because we just need hashable, ordered sets of small integers."""

    def __init__(self, numbers: Iterable[int] = None) -> None:
        self._bitset = 0
        if numbers is not None:
            for number in numbers:
                self._bitset |= 1 << number

    def union(self, other: "Bitset") -> "Bitset":
        result = Bitset()
        result._bitset = self._bitset | other._bitset
        return result

    def intersection(self, other: "Bitset") -> "Bitset":
        result = Bitset()
        result._bitset = self._bitset & other._bitset
        return result

    def minus(self, other: "Bitset") -> "Bitset":
        result = Bitset()
        result._bitset = self._bitset & ~other._bitset
        return result

    def is_empty(self) -> bool:
        return self._bitset == 0

    def indices(self, partial_bitset: "Bitset") -> list[int]:
        """Returns the indices of the partial bitset in the complete bitset. Example: `complete_bitset = 0b101101, partial_bitset = 0b100100 -> [1, 3]"""

        assert self & partial_bitset == partial_bitset, f"{bin(partial_bitset._bitset)} is not a subset of {bin(self._bitset)}"
        complete_bitset_copy = self._bitset
        partial_bitset_copy = partial_bitset._bitset
        indices = []
        current_index = 0
        while partial_bitset_copy:
            if complete_bitset_copy & 1:
                if partial_bitset_copy & 1:
                    indices.append(current_index)
                current_index += 1
            complete_bitset_copy >>= 1
            partial_bitset_copy >>= 1
        return indices

    def __or__(self, other: "Bitset") -> "Bitset":
        return self.union(other)

    def __and__(self, other: "Bitset") -> "Bitset":
        return self.intersection(other)

    def __sub__(self, other: "Bitset") -> "Bitset":
        return self.minus(other)

    def __contains__(self, element: int) -> bool:
        return (self._bitset >> element) & 1

    def __iter__(self) -> Iterable[int]:
        bitset_copy = self._bitset
        current_index = 0
        while bitset_copy:
            if bitset_copy & 1:
                yield current_index
            current_index += 1
            assert (bitset_copy >> 1) != bitset_copy, "We would go into an infinite loop here, aborting."
            bitset_copy >>= 1

    def __len__(self) -> int:
        return bin(self._bitset).count("1")

    def __eq__(self, other: "Bitset") -> bool:
        return self._bitset == other._bitset

    def __hash__(self) -> int:
        return hash(self._bitset)

    def __repr__(self) -> str:
        return f"{{{'' if self.is_empty() else ', '.join(str(i) for i in self)}}}"

    @staticmethod
    def full_set(n_variables: int) -> "Bitset":
        result = Bitset()
        result._bitset = (1 << n_variables) - 1
        return result

    @staticmethod
    def from_int(bitset: int) -> "Bitset":
        result = Bitset()
        result._bitset = bitset
        return result


def main():
    bitset_1 = Bitset.from_int(0b101101)
    bitset_2 = Bitset.from_int(0b100100)
    assert bitset_1.indices(bitset_2) == [1, 3]
    bitset_3 = Bitset.from_int(0b000100)
    bitset_4 = Bitset.from_int(0b101001)
    assert bitset_1 - bitset_3 == bitset_4
    assert list(bitset_1) == [0, 2, 3, 5]
    print(bin(Bitset.union(bitset_3, bitset_4)._bitset))


if __name__ == "__main__":
    main()
