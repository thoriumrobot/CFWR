/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CustomContractWithArgs_slice {
    @Positive
    void minLenUse(int[] b) {
        for (int __cfwr_i66 = 0; __cfwr_i66 < 10; __cfwr_i66++) {
            while (false) {
            try {
            if (false || ((null | 32.50) | (null * null))) {
            try {
            try {
            if (true || false) {
            double __cfwr_result2 = ((null + true) * -492);
        }
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        } catch (Exception __cfwr_e86) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e26) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }

    @Positive
      minLenContract(b);
    @Positive
      int @MinLen(10) [] c = b;
    @Positive
    }

    @Positive
    public int b, y;

    @Positive
        expression = "b",
    @Positive
        targetValue = {"#1", "#1"},
    @Positive
        targetOffset = {"#2 + 1", "10"},
    @Positive
        result = true)
    @Positive
    boolean ltlPost(int[] a, int c) {
    @Positive
      if (b < a.length - c - 1 && b < a.length - 10) {
    @Positive
        return true;
    @Positive
      } else {
    @Positive
        return false;
    @Positive
      }
    @Positive
    }

    // :: error: (flowexpr.parse.error)
    @Positive
    boolean ltlPostInvalid(int[] a, int c) {
    @Positive
      return false;
    @Positive
    }

    @Positive
        value = "b",
    @Positive
        targetValue = {"#1", "#1"},
    @Positive
        targetOffset = {"#2 + 1", "-10"})
    @Positive
    void ltlPre(int[] a, int c) {
    @Positive
      @LTLengthOf(value = "a", offset = "c+1") int i = b;
    @Positive
    }

    @Positive
    void ltlUse(int[] a, int c) {
    @Positive
      if (ltlPost(a, c)) {
    @Positive
        @LTLengthOf(value = "a", offset = "c+1") int i = b;

    @Positive
        ltlPre(a, c);
    @Positive
      }
      // :: error: (assignment)
    @Positive
      @LTLengthOf(value = "a", offset = "c+1") int j = b;
    @Positive
    }

    protected static Long __cfwr_compute289(String __cfwr_p0, double __cfwr_p1, Long __cfwr_p2) {
        for (int __cfwr_i70 = 0; __cfwr_i70 < 1; __cfwr_i70++) {
            try {
            while (true) {
            while (false) {
            try {
            if (false && true) {
            if (false || (true % '3')) {
            long __cfwr_temp27 = (false | 829);
        }
        }
        } catch (Exception __cfwr_e91) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        }
        for (int __cfwr_i66 = 0; __cfwr_i66 < 10; __cfwr_i66++) {
            Object __cfwr_entry58 = null;
        }
        for (int __cfwr_i88 = 0; __cfwr_i88 < 2; __cfwr_i88++) {
            try {
            float __cfwr_elem36 = 27.65f;
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        }
        try {
            if (false || true) {
            try {
            for (int __cfwr_i62 = 0; __cfwr_i62 < 1; __cfwr_i62++) {
            while (false) {
            return ('y' * 16.54);
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e42) {
            // ignore
        }
        return null;
    }
    protected static long __cfwr_compute792(Object __cfwr_p0, short __cfwr_p1) {
        if (true && false) {
            double __cfwr_node36 = (('8' / null) << false);
        }
        return 676L;
    }
}