import defaults as de
import numpy as np
from overlap import Overlap
import unittest
from dissimilarity import Dissimilarity
from GLV_model_class import Glv
import functions as fun
from DOC_class import DOC
from IDOA_class import IDOA

class TestOverlap(unittest.TestCase):
    """
    This class tests the Overlap.
    """
    def setUp(self) -> None:
        # Two default samples.
        self.sample1 = np.array([1, 0, 5, 7, 0, 3, 9, 2])
        self.sample2 = np.array([0, 0, 6, 2, 1, 0, 4, 0])
        # Initiations.
        self.o = Overlap(self.sample1, self.sample2)

    def test_normalization(self):
        """
        Normalization test
        """
        self.o.normalize()
        self.assertEqual(np.sum(self.o.normalized_sample_1), 1)
        self.assertEqual(np.sum(self.o.normalized_sample_2), 1)

    def test_non_zero(self):
        self.assertEqual(self.o.nonzero_index_1[0].tolist(), np.array([0, 2, 3, 5, 6, 7]).tolist())

    def test_s(self):
        self.assertEqual(self.o.S.tolist(), np.array([2, 3, 6]).tolist())
        self.assertEqual(first=self.o.normalized_sample_1[self.o.S].tolist()
                         , second=np.array([0.18518518518518517, 0.25925925925925924, 0.3333333333333333]).tolist())

    def test_solution(self):
        self.o.calculate_overlap()
        self.assertAlmostEqual(self.o.overlap, 0.8504273517, 5)

class TestDissimilarity(unittest.TestCase):
    """
    This class tests the Dissimilarity.
    """
    def setUp(self) -> None:
        # Two default samples.
        self.sample1 = np.array([1, 0, 5, 7, 0, 3, 9, 2])
        self.sample2 = np.array([0, 0, 6, 2, 1, 0, 4, 0])
        # Initiations.
        self.d = Dissimilarity(self.sample1, self.sample2)

    def test_calculate_normalised_in_s(self):
        test_arr = np.array([0.18518518518518517, 0.25925925925925924, 0.3333333333333333])
        self.assertEqual(self.d.normalized_sample_1_hat.tolist(), (test_arr/np.sum(test_arr)).tolist())

    def test_solution(self):
        self.d.calculate_dissimilarity()
        self.assertAlmostEqual(self.d.dissimilarity, 0.20221186759503793, 5)

class TestDefaults(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = de.calc_matrix(5)
        self.second_matrix = de.calc_matrix(5)
    def test_calc_matrix_vector_with_noise(self):
        no_noise_matrix = de.calc_matrix_vector_with_noise(10, self.matrix, noise_off=True)
        self.assertEqual(no_noise_matrix[1].tolist(), self.matrix.tolist())
        maximal_noise_matrix = de.calc_matrix_vector_with_noise(10, self.matrix, maximum_noise=True)
        self.assertNotEqual(maximal_noise_matrix[0].tolist(), maximal_noise_matrix[1].tolist())

    def test_calc_matrix_vector_with_noise_non_homogeneous(self):
        no_noise_matrix = de.calc_matrix_vector_with_noise_non_homogeneous(
            10, self.matrix, self.second_matrix, noise_off=True, maximum_noise=False)
        self.assertEqual(no_noise_matrix[4].tolist(), self.matrix.tolist())
        self.assertEqual(no_noise_matrix[6].tolist(), self.second_matrix.tolist())
        maximal_noise_matrix = de.calc_matrix_vector_with_noise_non_homogeneous(
            10, self.matrix, self.second_matrix, noise_off=False, maximum_noise=True)
        self.assertNotEqual(maximal_noise_matrix[4].tolist(), maximal_noise_matrix[6].tolist())

class TestGlv(unittest.TestCase):
    def setUp(self) -> None:
        self.number_of_species = 10
        self.s = np.ones(self.number_of_species)
        self.r = np.random.uniform(0, 1, (self.number_of_species,))
        self.delta = 10 ** -4
        self.number_of_samples = 10
        self.time_span = 10
        self.max_step = 3
        self.interact_mat = de.calc_matrix(self.number_of_species)
        self.init_cond = de.clac_set_of_initial_conditions(self.number_of_species, self.number_of_samples)
        self.glv_model = Glv(self.number_of_samples, self.number_of_species, self.delta, self.r, self.s, self.
                             interact_mat, self.init_cond, self.time_span, self.max_step)

    def test_normalization(self):
        self.glv_model.solve()
        self.glv_model.normalize_results()
        final_abundances = self.glv_model.norm_Final_abundances
        self.assertEqual((np.sum(final_abundances, axis=1)).tolist(), np.ones(self.number_of_samples).tolist())

class TestGraphs1(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = np.array([0, 1, 4, 0, 5])
        self.cohort = np.array([[2, 3, 0, 1, 0], [0, 2, 1, 3, 7], [10, 5, 2, 1, 6], [2, 1, 0, 0, 6]])
        self.overlap_vector = np.zeros(np.size(self.cohort, axis=0))
        self.dissimilarity_vector = np.zeros(np.size(self.cohort, axis=0))

    def test_IDOA(self):
        [new_overlap_vector, new_dissimilarity_vector] = IDOA(self.sample, self.cohort,
                                                              self.overlap_vector, self.dissimilarity_vector)
        self.assertGreater(new_overlap_vector.tolist(), [0.5, 0.5, 0.5, 0.5])

class TestGraphs3(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.array([[1, 2, 0, 7], [0, 3, 5, 0], [2, 3, 8, 1], [9, 0, 6, 6]])

class TestG3(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.array([[1, 0, 4, 5], [3, 2, 6, 0], [9, 0, 0, 0], [3, 4, 0, 1], [2, 2, 4, 18]])
        self.single_sample = np.array([0, 2, 0, 4, 2])

    def test_find_most_abundant_rows(self):
        most_ab_data = fun.find_most_abundant_rows(self.data, 1)
        self.assertEqual(most_ab_data.tolist(), np.array([[2, 2, 4, 18]]).tolist())

    def test_normalize_data(self):
        norm_data = fun.normalize_data(self.data)
        self.assertEqual(np.sum(norm_data, axis=0).tolist(), np.ones(np.size(self.data, axis=1)).tolist())

    def test_create_shuffled(self):
        shuffled = fun.create_shuffled(self.single_sample, self.data)
        self.assertEqual(shuffled[2], 0)

    def test_create_randomized_cohort(self):
        self.data = np.array([[1, 0, 4, 5], [3, 2, 6, 0], [3, 0, 4, 5]]).T
        self.samples = np.array([[1, 0, 4, 5], [3, 2, 6, 0]]).T
        rand_samples = fun.create_randomized_cohort(self.data, self.samples)

class TestFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.first_cohort = np.array([[1, 2, 5, 8], [3, 6, 8, 0], [5, 2, 1, 4]])
        self.second_cohort = np.array([[5, 7, 9, 3], [4, 4, 6, 1], [5, 0, 2, 1], [2, 3, 6, 1]])

    def test_calc_bray_curtis_dissimilarity(self):
        self.assertEqual(np.size(fun.calc_bray_curtis_dissimilarity(self.first_cohort, self.second_cohort)),
                         np.size(self.second_cohort, axis=0))

        self.assertEqual(fun.calc_bray_curtis_dissimilarity_sklearn(self.first_cohort).tolist(),
                         fun.calc_bray_curtis_dissimilarity(self.first_cohort, self.first_cohort).tolist())

class TestDoc(unittest.TestCase):
    def setUp(self) -> None:
        self.cohort = np.array([[1, 2, 2, 1, 1], [1, 2, 0, 1, 1], [2, 1, 2, 2, 1], [2, 2, 2, 2, 3]])
        self.doc = DOC(self.cohort)

    def test_Doc(self):
        DOC_mat = self.doc.calc_DOC()
        print(DOC_mat)

class TestIdoaClass(unittest.TestCase):
    """
    This class tests the IDOA class.
    """
    def setUp(self) -> None:
        # Define simple normalized cohorts.
        self.cohort_1 = np.array([[0.2, 0, 0.4, 0, 0.4], [0, 0.1, 0.3, 0.2, 0.4], [0.5, 0.4, 0.05, 0.05, 0],
                                 [0.1, 0.1, 0.1, 0.1, 0.6]])
        self.cohort_2 = np.array([[0.1, 0.1, 0.5, 0, 0.3], [0.2, 0, 0.4, 0.1, 0.3], [0.3, 0.4, 0, 0.1, 0.2],
                                 [0.2, 0.2, 0.2, 0.1, 0.3]])
        # Initialize IDOA not counting for the fact we loaded the same cohort.
        self.IDOA_self_cohort = IDOA(self.cohort_1, self.cohort_1, self_cohort=False)
        # Initialize IDOA counting for the fact we loaded the same cohort, this means we're not calculating
        # Dissimilarity and overlap of the same sample.
        self.IDOA_self_cohort_True = IDOA(self.cohort_1, self.cohort_1, self_cohort=True)

    def test_IDOA_class(self):
        # Apply IDOA for both cases.
        self.IDOA_vector_self_cohort = self.IDOA_self_cohort.calc_idoa_vector()
        self.IDOA_vector_self_cohort_True = self.IDOA_self_cohort_True.calc_idoa_vector()
        # Here I check if the diagonal of the overlap and dissimilarity matrices have ones and zeros respectively on the
        # diagonal, I expect it to be True.
        self.assertEqual(np.ones(self.cohort_1.shape[0]).tolist(), np.diag(self.IDOA_self_cohort.overlap_mat).tolist())
        self.assertEqual(np.zeros(self.cohort_1.shape[0]).tolist(), np.diag(self.IDOA_self_cohort.dissimilarity_mat
                                                                            ).tolist())
        # Here I check the opposite case.
        self.assertNotEqual(np.ones(self.cohort_1.shape[0]).tolist(), np.diag(
            self.IDOA_self_cohort_True.overlap_mat).tolist())
        self.assertNotEqual(np.zeros(self.cohort_1.shape[0]).tolist(), np.diag(
            self.IDOA_self_cohort_True.dissimilarity_mat).tolist())

        # Here I reshape the overlap matrices to two vectors, the first contains the off diagonal elements of the first
        # case overlap matrix, the second contains all the elements of the second case overlap vector.
        mask = np.ones_like(self.IDOA_self_cohort.overlap_mat, dtype=bool)
        mask[range(self.IDOA_self_cohort.overlap_mat.shape[0]), range(
            self.IDOA_self_cohort.overlap_mat.shape[1])] = False
        overlap_vector = self.IDOA_self_cohort.overlap_mat[mask]
        overlap_vector_True = self.IDOA_self_cohort_True.overlap_mat.reshape(-1)

        # Here I test If all the elements in these vectors are equal.
        self.assertEqual(overlap_vector.tolist(), overlap_vector_True.tolist())

        # Same process for the dissimilarity matrices.
        mask = np.ones_like(self.IDOA_self_cohort.dissimilarity_mat, dtype=bool)
        mask[range(self.IDOA_self_cohort.dissimilarity_mat.shape[0]), range(
            self.IDOA_self_cohort.dissimilarity_mat.shape[1])] = False
        dissimilarity_vector = self.IDOA_self_cohort.dissimilarity_mat[mask]
        dissimilarity_vector_True = self.IDOA_self_cohort_True.dissimilarity_mat.reshape(-1)

        self.assertEqual(dissimilarity_vector.tolist(), dissimilarity_vector_True.tolist())

        print(self.IDOA_self_cohort.overlap_mat)
        print(self.IDOA_self_cohort.dissimilarity_mat)
        print(self.IDOA_self_cohort_True.overlap_mat)
        print(self.IDOA_self_cohort_True.dissimilarity_mat)
