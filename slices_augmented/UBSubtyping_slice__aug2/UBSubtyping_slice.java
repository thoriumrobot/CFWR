/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class UBSubtyping_slice {
    @Positive
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
        if (false || false) {
            for (int __cfwr_i19 = 0; __cfwr_i19 < 3; __cfwr_i19++) {
            return null;
        }
        }

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int a = 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a1 = 1;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int b = a;
    @Positive
    @UpperBoundUnknown int d = a;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr2"}) int g = a;

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr", "arr2", "arr3"}) int h = 2;

    @Positive
    @LTEqLengthOf({"arr", "arr2"}) int h2 = test;
    @Positive
    @LTEqLengthOf({"arr"}) int i = test;
    @
        if (true && true) {
            while (('K' & (-63.43 % 'C'))) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 2; __cfwr_i14++) {
            return -63.85f;
        }
            break; // Prevent infinite loops
        }
        }
Positive
    @LTEqLengthOf({"arr", "arr3"}) int j = test;
    @Positive
  }

    static Float __cfwr_temp329(Integer __cfwr_p0, byte __cfwr_p1) {
        if (('W' >> null) || false) {
            while (((554L - false) * true)) {
            if (true && false) {
            try {
            for (int __cfwr_i39 = 0; __cfwr_i39 < 2; __cfwr_i39++) {
            return null;
        }
        } catch (Exception __cfwr_e7) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i52 = 0; __cfwr_i52 < 9; __cfwr_i52++) {
            return false;
        }
        return -27L;
        return null;
    }
}