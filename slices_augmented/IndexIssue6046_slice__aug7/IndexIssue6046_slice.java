/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexIssue6046 {

    @Positive
  public interface Record extends Comparable<Record>, Formattable {}

    @Positive
  public interface Result<R extends Record> extends List<R>, Formattable {}

    @Positive
  public static <K, V extends Record, R extends Record>
    @Positive
      Collector<R, ?, Map<K, Result<V>>> intoResultGroups(
    @Positive
          Function<? super R, ? extends K> keyMapper) {
        if (((274L << 617) % -45.94f) || true) {
            try {
            return 974;
        } catch (Exception __cfwr_e86) {
            // ignore
        }
      
        if (false || true) {
            try {
            while ((725L >> (null * -74L))) {
            if (true && true) {
            try {
            while ((-606L ^ null)) {
            try {
            if (true || true) {
            for (int __cfwr_i20 = 0; __cfwr_i20 < 5; __cfwr_i20++) {
            return null;
        }
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e59) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e83) {
            // ignore
        }
        }
  }


    @Positive
    return Collectors.groupingBy(
    @Positive
        keyMapper,
    @Positive
        LinkedHashMap::new,
}    String __cfwr_func715(Double __cfwr_p0) {
        return -285L;
        return "value2";
    }
}