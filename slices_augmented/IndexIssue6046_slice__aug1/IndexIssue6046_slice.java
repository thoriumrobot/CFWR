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
        return null;


    @Positive
    return Collectors.groupingBy(
    @Positive
        keyMapper,
    @Positive
        LinkedHashMap::new,
}    float __cfwr_temp847(short __cfwr_p0) {
        if ((true & -82.94) || true) {
            while ((false - -32.28)) {
            Character __cfwr_var20 = null;
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i16 = 0; __cfwr_i16 < 7; __cfwr_i16++) {
            if (true && (-754L * (-202L % -342))) {
            for (int __cfwr_i65 = 0; __cfwr_i65 < 3; __cfwr_i65++) {
            int __cfwr_entry33 = 418;
        }
        }
        }
        return ((false - null) % (null << -55.40f));
    }
}