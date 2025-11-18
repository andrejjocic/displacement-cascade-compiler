import unittest
from dsdobjects import DomainS, ComplexS
import sys
import os
import hashlib
import random
import string
import time

sys.path.append("src") # NOTE: assuming test will be run from root!
from utils import SecondaryStructure
from compile import compile_LTA_circuit


def random_domain(length=5):
    """Generate a random domain with a unique, readable name.
    
    Uses a hash of current time and random value to create a unique name.
    """
    # Create a unique hash based on time and random value
    unique_hash = hashlib.md5(f"{time.time()}_{random.random()}".encode()).hexdigest()[:10]
    
    # Create a readable prefix (1-3 letters)
    prefix_length = random.randint(1, 3)
    prefix = ''.join(random.choices(string.ascii_lowercase, k=prefix_length))
    
    # Combine prefix with hash to create a unique name
    domain_name = f"{prefix}_{unique_hash}"
    
    return DomainS(domain_name, length)


class TestSecondaryStructure(unittest.TestCase):

    def setUp(self):
        # Create common domains for testing
        self.domain_a = DomainS("a", 5)
        self.domain_b = DomainS("b", 5)
        self.domain_c = DomainS("c", 5)
        self.domain_d = DomainS("d", 5)
        
        self.lta_circ = compile_LTA_circuit(3)


    def assert_nonPK_complex(self, cpx: ComplexS):
        """all ComplexS should be non-pseudoknotted"""
        strands = list(cpx.strand_table)
        pairs = list(cpx.pair_table)

        ss = SecondaryStructure({f"s{i}": strand for i, strand in enumerate(strands)})

        for i in range(len(strands)):
            for j, entry in enumerate(pairs[i]):
                if entry is None: continue # non-hybridized domain
                k, l = entry

                if (i, j) < (k, l):
                    ss.hybridize(f"s{i}", j, f"s{k}", l)
                else:
                    self.assertEqual(ss.bindings[(f"s{i}", j)], (f"s{k}", l), 
                                     f"pair table reverse link should already exist for {i}, {j} to {k}, {l}")
                    # NOTE: we sure this is guaranteed by this traversal order?

        self.assertFalse(ss.pseudoknotted)


    def test_nonPK_circuit(self):
        for cpx in self.lta_circ._init_rel_conc.keys():
            self.assert_nonPK_complex(cpx)


    def test_single_strand_structure(self):
        """Test a structure with a single strand."""
        # A single strand cannot form pseudoknots
        strands = {"s1": [self.domain_a, self.domain_b, self.domain_c]}
        structure = SecondaryStructure(strands)
        
        # No bindings, not pseudoknotted
        self.assertFalse(structure.pseudoknotted)

    
    def test_ssDNA_looped_nonPK(self):
        # 2014 peppercorn paper fig. S4a
        blue, cyan, orange, green, vomit = (random_domain() for _ in range(5))
        structure = SecondaryStructure({"": [blue, cyan, orange, ~cyan, vomit, green, ~vomit]})
        structure.hybridize("", 1, "", 1+2)  # cyan helix
        structure.hybridize("", -1, "", -3) # vomit helix
        self.assertFalse(structure.pseudoknotted)


    def test_simple_pseudoknot(self):
        # 2014 peppercorn paper fig. S4b
        dBlue5, lBlue, dOrange, lOrange, green, dBlue3 = (random_domain() for _ in range(6))
        pk = SecondaryStructure({
            "s": [dBlue5, lBlue, dOrange, lOrange, ~lBlue, green, ~lOrange, dBlue3]
        })
        pk.hybridize("s", 1, "s", 4)  # light blue helix
        pk.hybridize("s", 3, "s", 6)  # light orange helix
        self.assertTrue(pk.pseudoknotted)


    # def test_all_canonical_loops(self):
    #     # peppercorn paper intro big example (also 2014 paper fig. S3)
    #     #cpx = SecondaryStructure({"red": red, "blue": blue, "green": green, "pink": pink, "orange": orange})
    #     cpx = SecondaryStructure.from_kernel_notation("...")
    #     self.assertFalse(cpx.pseudoknotted)


    def test_kissing_loops(self):
        # 2014 peppercorn paper fig. S4d
        gray, cyan, blue, orange = (random_domain() for _ in range(4))
        s_left = [gray, cyan, blue, orange, ~gray] # clockwise
        s_right = [gray, ~orange, ~blue, ~cyan, ~gray] # CCW
        kiss = SecondaryStructure({"left": s_left, "right": s_right})
        
        kiss.hybridize("left", 2, "right", 2) # kiss at (dark) blue
        kiss.hybridize("left", 0, "left", -1) # close left stem
        kiss.hybridize("right", 0, "right", -1) # close right stem

        self.assertTrue(kiss.pseudoknotted)



if __name__ == "__main__":
    unittest.main()
