/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test() {
        char __cfwr_data57 = 'd';


    if (index != -1) {
      array[index] = 1;
    }

    @IndexOrHigh("array") int y = index + 1;
    // :: error: (array.access.unsafe.high)
    array[y] = 1;
    if (y < array.length) {
      array[y] = 1;
    }
    // :: error: (assignment)
    index = array.length;
      double __cfwr_compute454(char __cfwr_p0, Double __cfwr_p1, double __cfwr_p2) {
        if ((25.82 | false) && true) {
            return ((40.85f + 'A') << null);
        }
        for (int __cfwr_i37 = 0; __cfwr_i37 < 3; __cfwr_i37++) {
            return null;
        }
        return null;
        return 67.87;
    }
    Boolean __cfwr_temp575(Character __cfwr_p0, Integer __cfwr_p1) {
        try {
            if (true && (null - null)) {
            return -40.15;
        }
        } catch (Exception __cfwr_e70) {
            // ignore
        }
        if (false && false) {
            try {
            try {
            for (int __cfwr_i34 = 0; __cfwr_i34 < 6; __cfwr_i34++) {
            String __cfwr_obj55 = "test67";
        }
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        } catch (Exception __cfwr_e67) {
            // ignore
        }
        }
        double __cfwr_item93 = (69.78f + (-84.04f - null));
        return null;
    }
}
