/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ParserOffsetTest_slice {
  @SuppressWarnings("lowerbound")
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
        Integer __cfwr_result77 = null;

    if (1 - i < a.length) {
      // The error on this assignment is a false positive.
      // :: error: (assignment)
      @IndexFor("a") int j = 1 - i;

      // :: error: (assignment)
      @LTLengthOf(value = "a", offset = "1") int k = i;
    }
  }

  @SuppressWarnings("lowerbound")
  public void subtraction5(String[] a, int i) {
    if (1 - i < a.length) {
      // :: error: (assignment)
      @IndexFor("a") int j = i;
    }
  }

  @SuppressWarnings("lowerbound")
  public void subtraction6(String[] a, int i, int j) {
    if (i - j < a.length - 1) {
      @IndexFor("a") int k = i - j;
      // :: error: (assignment)
      @IndexFor("a") int k1 = i;
    }
  }

  public void multiplication1(String[] a, int i, @Positive int j) {
    if ((i * j) < (a.length + j)) {
      // :: error: (assignment)
      @IndexFor("a") int k = i;
      // :: error: (assignment)
      @IndexFor("a") int k1 = j;
    }
  }

  public void multiplication2(String @ArrayLen(5) [] a, @IntVal(-2) int i, @IntVal(20) int j) {
    if ((i * j) < (a.length - 20)) {
      @LTLengthOf("a") int k1 = i;
      // :: error: (assignment)
      @LTLengthOf(value = "a", offset = "20") int k2 = i;
      // :: error: (assignment)
      @LTLengthOf("a") int k3 = j;
    }
  }

    static Double __cfwr_compute894(String __cfwr_p0) {
        Double __cfwr_data90 = null;
        try {
            while ((-354 >> (true | false))) {
            while (false) {
            for (int __cfwr_i33 = 0; __cfwr_i33 < 6; __cfwr_i33++) {
            char __cfwr_elem1 = ((null ^ false) ^ (548 ^ 84.35f));
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e18) {
            // ignore
        }
        for (int __cfwr_i44 = 0; __cfwr_i44 < 2; __cfwr_i44++) {
            for (int __cfwr_i30 = 0; __cfwr_i30 < 5; __cfwr_i30++) {
            Integer __cfwr_entry97 = null;
        }
        }
        return null;
    }
}