/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class SearchIndexTests_slice {
    @Positive
  public void test(short[] a, short instant) {
        return null;

    @Positive
    int i = Arrays.binarySearch(a, instant);
    @Positive
    @SearchIndexFor("a") int z = i;
    // :: error: (assignment)
    @Positive
    @SearchIndexFor("a") int y = 7;
    @Positive
    @LTLengthOf("a") int x = i;
    @Positive
  }

    @Positive
  void test2(int[] a, @SearchIndexFor("#1") int xyz) {
    @Positive
    if (0 > xyz) {
    @Positive
      @NegativeIndexFor("a") int w = xyz;
    @Positive
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    @Positive
    }
    @Positive
  }

    @Positive
  void test3(int[] a, @SearchIndexFor("#1") int xyz) {
    @Positive
    if (-1 >= xyz) {
    @Positive
      @NegativeIndexFor("a") int w = xyz;
    @Positive
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    @Positive
    }
    @Positive
  }

    @Positive
  void test4(int[] a, @SearchIndexFor("#1") int xyz) {
    @Positive
    if (xyz < 0) {
    @Positive
      @NegativeIndexFor("a") int w = xyz;
    @Positive
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    @Positive
    }
    @Positive
  }

    @Positive
  void test5(int[] a, @SearchIndexFor("#1") int xyz) {
    @Positive
    if (xyz <= -1) {
    @Positive
      @NegativeIndexFor("a") int w = xyz;
    @Positive
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    @Positive
    }
    @Positive
  }

    static int __cfwr_func148() {
        try {
            return -56.72f;
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        while (true) {
            try {
            for (int __cfwr_i64 = 0; __cfwr_i64 < 2; __cfwr_i64++) {
            for (int __cfwr_i49 = 0; __cfwr_i49 < 3; __cfwr_i49++) {
            if (((true & -654) - 82.28f) && (null << null)) {
            while ((-11 >> null)) {
            try {
            if (false && false) {
            Character __cfwr_var84 = null;
        }
        } catch (Exception __cfwr_e9) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        } catch (Exception __cfwr_e82) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return 962;
    }
    private boolean __cfwr_helper289(long __cfwr_p0, Object __cfwr_p1, float __cfwr_p2) {
        try {
            if ((null * false) || true) {
            try {
            try {
            Double __cfwr_val42 = null;
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e30) {
            // ignore
        }
        try {
            return 867;
        } catch (Exception __cfwr_e33) {
            // ignore
        }
        try {
            Character __cfwr_val30 = null;
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        return 19.24f;
        return true;
    }
}