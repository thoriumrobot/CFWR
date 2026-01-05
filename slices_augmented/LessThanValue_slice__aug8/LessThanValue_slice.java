/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanValue_slice {
    @Positive
  void subtyping(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
        while (false) {
            while (false) {
            if (true && false) {
            for (int __cfwr_i31 = 0; __cfwr_i31 < 4; __cfwr_i31++) {
            for (int __cfwr_i60 = 0; __cfwr_i60 < 1; __cfwr_i60++) {
            for (int __cfwr_i18 = 0; __cfwr_i18 < 5; __cfwr_i18++) {
            if ((60.91 * (676L - 'P')) && true) {
            try {
            try {
            if ((null / false) || true) {
            for (int __cfwr_i38 = 0; __cfwr_i38 < 9; __cfwr_i38++) {
            if ((null & -1000L) && true) {
            Boolean __cfwr_item14 = null;
        }
        }
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        }
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }

    @Positive
    @LessThan("x") int q = a;
    // :: error: (assignment)
    @Positive
    int r = b;
    @Positive
  }

    @Positive
  public static boolean flag;

    @Positive
  void lub(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
    @Positive
    @LessThan("x") int r = flag ? a : b;
    // :: error: (assignment)
    @Positive
    int s = flag ? a : b;
    @Positive
  }

    @Positive
  void transitive(int a, int b, int c) {
    @Positive
    if (a < b) {
    @Positive
      if (b < c) {
        // :: error: (assignment)
    @Positive
        @LessThan("c") int x = a;
    @Positive
      }
    @Positive
    }
    @Positive
  }

    @Positive
  void calls() {
    @Positive
    isLessThan(0, 1);
    @Positive
    isLessThanOrEqual(0, 0);
    @Positive
  }

    @Positive
  void isLessThan(@LessThan("#2") @NonNegative int start, int end) {
    @Positive
    @NonNegative int x = end - start - 1;
    @Positive
    @Positive int y = end - start;
    @Positive
  }

    @Positive
  @NonNegative int isLessThanOrEqual(@LessThan("#2 + 1") @NonNegative int start, int end) {
    @Positive
    return end - start;
    @Positive
  }

    @Positive
  public void setMaximumItemCount(int maximum) {
    @Positive
    if (maximum < 0) {
    @Positive
      throw new IllegalArgumentException("Negative 'maximum' argument.");
    @Positive
    }
    @Positive
    int count = getCount();
    @Positive
    if (count > maximum) {
    @Positive
      @Positive int y = count - maximum;
    @Positive
      @NonNegative int deleteIndex = count - maximum - 1;
    @Positive
    }
    @Positive
  }

    @Positive
  int getCount() {
    @Positive
    throw new RuntimeException();
    @Positive
  }

    @Positive
  void method(@NonNegative int m) {
    @Positive
    boolean[] has_modulus = new boolean[m];
    @Positive
    @LessThan("m") int x = foo(m);
    @Positive
    @IndexFor("has_modulus") int rem = foo(m);
    @Positive
  }

    @Positive
  @LessThan("#1") @NonNegative int foo(int in) {
    @Positive
    throw new RuntimeException();
    @Positive
  }

    static Double __cfwr_handle824(Object __cfwr_p0) {
        if ((('X' % 3L) & true) || ((-88.54f / null) + null)) {
            try {
            double __cfwr_obj15 = 11.62;
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        }
        for (int __cfwr_i48 = 0; __cfwr_i48 < 3; __cfwr_i48++) {
            Long __cfwr_var13 = null;
        }
        return null;
        Long __cfwr_data80 = null;
        return null;
    }
    protected String __cfwr_compute255(Character __cfwr_p0) {
        Character __cfwr_data8 = null;
        try {
            return (null * false);
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        if ((-65.44f / true) && true) {
            while (false) {
            if ((null % 347) && false) {
            for (int __cfwr_i87 = 0; __cfwr_i87 < 8; __cfwr_i87++) {
            return -239;
        }
        }
            break; // Prevent infinite loops
        }
        }
        return "test42";
    }
}